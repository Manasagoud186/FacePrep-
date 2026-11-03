import os
import sys
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Dict, Any
import json
from datetime import datetime, timezone
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)


from database import engine, Base, get_db
import models
import schemas
from face_processor import process_frame, process_face_samples, verify_face_in_frame
from questions import QUESTIONS, get_questions_for_session
from speech_analyzer import analyze_answer
from gaze_analyzer import analyze_gaze_frames
from answer_analyzer import analyze_answer_concepts, DISCLAIMER_TEXT

# Create tables and run column migrations on startup
Base.metadata.create_all(bind=engine)
try:
    from migrate_db import run_migrations
    run_migrations()
except Exception as _mig_err:
    print(f"[startup] Migration warning: {_mig_err}")

app = FastAPI(title="FacePrep API", description="Backend for FacePrep Mock Interview System")


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request, exc):
    """Return a stable API error for database outages without leaking SQL details."""
    print(f"[database] Request failed: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "The database is temporarily unavailable. Please retry."},
    )

# Configure CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to FacePrep API"}

@app.get("/api/health")
def health_check():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok", "message": "Backend and database are running"}

@app.post("/api/analyze-frame", response_model=schemas.FrameAnalysisResponse)
def analyze_frame(payload: schemas.FrameAnalysisRequest):
    """
    Analyze a single frame from the live webcam feed for face detection,
    position guidance, and count check.
    """
    result = process_frame(payload.image)
    return schemas.FrameAnalysisResponse(
        status=result["status"],
        message=result["message"],
        face_count=result["face_count"],
        guidance=result["guidance"],
        bounding_box=result["bounding_box"]
    )

@app.post("/api/verify", response_model=schemas.FaceVerificationResponse)
def verify_identity(payload: schemas.FaceVerificationRequest, db: Session = Depends(get_db)):
    """
    Verify candidate identity against all registered users' facial embeddings.
    Extracts facial representation using the exact same feature algorithm as registration.
    """
    # Fetch all stored face embeddings from database
    users = db.query(models.User).all()
    registered_embeddings = []
    user_map = {}

    for user in users:
        user_map[user.id] = user
        if user.face_data and user.face_data.embedding:
            try:
                emb = json.loads(user.face_data.embedding)
                registered_embeddings.append((user.id, emb))
            except Exception as e:
                print(f"Error parsing face embedding for user {user.id}: {e}")
                continue

    result = verify_face_in_frame(payload.image, registered_embeddings, threshold=0.70)

    matched_user_resp = None
    if result.get("matched_user_id") and result["matched_user_id"] in user_map:
        matched_user = user_map[result["matched_user_id"]]
        has_face = matched_user.face_data is not None
        face_resp = None
        if has_face:
            face_resp = schemas.FaceDataResponse(
                id=matched_user.face_data.id,
                user_id=matched_user.face_data.user_id,
                sample_count=matched_user.face_data.sample_count,
                created_at=matched_user.face_data.created_at
            )
        matched_user_resp = schemas.UserResponse(
            id=matched_user.id,
            full_name=matched_user.full_name,
            student_id=matched_user.student_id,
            email=matched_user.email,
            target_role=matched_user.target_role,
            experience_level=matched_user.experience_level,
            created_at=matched_user.created_at,
            has_face_data=has_face,
            face_data=face_resp
        )

    return schemas.FaceVerificationResponse(
        status=result["status"],
        message=result["message"],
        face_count=result["face_count"],
        confidence=result.get("confidence"),
        user=matched_user_resp,
        bounding_box=result.get("bounding_box")
    )


@app.post("/api/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new candidate profile, process face samples, extract feature embedding vector,
    and save both in SQLite database.
    """
    # Check if student_id or email already exists
    existing_student = db.query(models.User).filter(models.User.student_id == user_data.student_id).first()
    if existing_student:
        raise HTTPException(
            status_code=400,
            detail=f"Student ID '{user_data.student_id}' is already registered."
        )
        
    existing_email = db.query(models.User).filter(models.User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=400,
            detail=f"Email address '{user_data.email}' is already registered."
        )

    # Ensure face samples were provided
    if not user_data.face_samples or len(user_data.face_samples) < 1:
        raise HTTPException(
            status_code=400,
            detail="Face samples are required for registration. Please capture your face."
        )

    # Process face samples and extract feature embedding
    success, msg, embedding, valid_sample_count = process_face_samples(user_data.face_samples)
    if not success or embedding is None:
        raise HTTPException(
            status_code=400,
            detail=f"Face registration failed: {msg}"
        )

    # Save profile and representation atomically; never leave a partial user if
    # the face record cannot be committed.
    try:
        db_user = models.User(
            full_name=user_data.full_name,
            student_id=user_data.student_id,
            email=user_data.email,
            target_role=user_data.target_role,
            experience_level=user_data.experience_level
        )
        db.add(db_user)
        db.flush()

        db_face_data = models.FaceData(
            user_id=db_user.id,
            embedding=json.dumps(embedding),
            sample_count=valid_sample_count
        )
        db.add(db_face_data)
        db.commit()
        db.refresh(db_user)
        db.refresh(db_face_data)
    except SQLAlchemyError:
        db.rollback()
        raise

    return schemas.UserResponse(
        id=db_user.id,
        full_name=db_user.full_name,
        student_id=db_user.student_id,
        email=db_user.email,
        target_role=db_user.target_role,
        experience_level=db_user.experience_level,
        created_at=db_user.created_at,
        has_face_data=True,
        face_data=schemas.FaceDataResponse(
            id=db_face_data.id,
            user_id=db_face_data.user_id,
            sample_count=valid_sample_count,
            created_at=db_face_data.created_at
        )
    )

@app.get("/api/users", response_model=List[schemas.UserResponse])
def get_users(db: Session = Depends(get_db)):
    """List all registered candidate profiles."""
    users = db.query(models.User).all()
    response = []
    for u in users:
        has_face = u.face_data is not None
        face_resp = None
        if has_face:
            face_resp = schemas.FaceDataResponse(
                id=u.face_data.id,
                user_id=u.face_data.user_id,
                sample_count=u.face_data.sample_count,
                created_at=u.face_data.created_at
            )
        response.append(schemas.UserResponse(
            id=u.id,
            full_name=u.full_name,
            student_id=u.student_id,
            email=u.email,
            target_role=u.target_role,
            experience_level=u.experience_level,
            created_at=u.created_at,
            has_face_data=has_face,
            face_data=face_resp
        ))
    return response

@app.get("/api/users/{user_id}", response_model=schemas.UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a specific registered candidate profile by ID."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    has_face = user.face_data is not None
    face_resp = None
    if has_face:
        face_resp = schemas.FaceDataResponse(
            id=user.face_data.id,
            user_id=user.face_data.user_id,
            sample_count=user.face_data.sample_count,
            created_at=user.face_data.created_at
        )
        
    return schemas.UserResponse(
        id=user.id,
        full_name=user.full_name,
        student_id=user.student_id,
        email=user.email,
        target_role=user.target_role,
        experience_level=user.experience_level,
        created_at=user.created_at,
        has_face_data=has_face,
        face_data=face_resp
    )

@app.delete("/api/users/{user_id}/face-data")
def delete_face_data(user_id: int, db: Session = Depends(get_db)):
    """Delete stored facial feature representation / embeddings for a user."""
    face_record = db.query(models.FaceData).filter(models.FaceData.user_id == user_id).first()
    if not face_record:
        raise HTTPException(status_code=404, detail="No face data found for this user.")
        
    db.delete(face_record)
    db.commit()
    return {"status": "success", "message": "Face data successfully deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/questions", response_model=List[schemas.QuestionSchema])
def get_all_questions():
    """Return the full question bank (for inspection)."""
    return [schemas.QuestionSchema(**q) for q in QUESTIONS]


@app.post("/api/interview/session", response_model=schemas.InterviewSessionResponse, status_code=status.HTTP_201_CREATED)
def create_interview_session(payload: schemas.InterviewSessionCreate, db: Session = Depends(get_db)):
    """
    Create a new interview session for a verified user.
    Selects 5 questions matching the chosen role, type, and difficulty.
    """
    user = db.query(models.User).filter(models.User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Clamp question count
    q_count = max(1, min(payload.question_count, 10))

    selected_questions = get_questions_for_session(
        role=payload.role,
        interview_type=payload.interview_type,
        difficulty=payload.difficulty,
        count=q_count
    )

    if not selected_questions:
        raise HTTPException(status_code=422, detail="No questions available for the selected configuration.")

    question_ids = [q["id"] for q in selected_questions]

    session = models.InterviewSession(
        user_id=payload.user_id,
        role=payload.role,
        interview_type=payload.interview_type,
        difficulty=payload.difficulty,
        question_ids=json.dumps(question_ids),
        status="in_progress",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return schemas.InterviewSessionResponse(
        id=session.id,
        user_id=session.user_id,
        role=session.role,
        interview_type=session.interview_type,
        difficulty=session.difficulty,
        status=session.status,
        question_ids=question_ids,
        questions=[schemas.QuestionSchema(**q) for q in selected_questions],
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


@app.get("/api/interview/session/{session_id}", response_model=schemas.InterviewSessionResponse)
def get_interview_session(session_id: int, db: Session = Depends(get_db)):
    """Retrieve an existing interview session with its questions."""
    session = db.query(models.InterviewSession).filter(models.InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    q_ids: List[int] = json.loads(session.question_ids)
    q_map = {q["id"]: q for q in QUESTIONS}
    questions = [schemas.QuestionSchema(**q_map[qid]) for qid in q_ids if qid in q_map]

    return schemas.InterviewSessionResponse(
        id=session.id,
        user_id=session.user_id,
        role=session.role,
        interview_type=session.interview_type,
        difficulty=session.difficulty,
        status=session.status,
        question_ids=q_ids,
        questions=questions,
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


@app.get("/api/interview/session/{session_id}/answers", response_model=List[schemas.AnswerResponse])
def get_session_answers(session_id: int, db: Session = Depends(get_db)):
    """Retrieve all recorded answers and analysis results for a session."""
    session = db.query(models.InterviewSession).filter(models.InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    answers = db.query(models.InterviewAnswer).filter(models.InterviewAnswer.session_id == session_id).all()
    resp = []
    for a in answers:
        detected = json.loads(a.detected_concepts) if a.detected_concepts else None
        missing = json.loads(a.missing_concepts) if a.missing_concepts else None
        explanations = json.loads(a.concept_explanations) if a.concept_explanations else None

        resp.append(schemas.AnswerResponse(
            id=a.id,
            session_id=a.session_id,
            question_id=a.question_id,
            question_text=a.question_text,
            audio_duration_seconds=a.audio_duration_seconds,
            recorded_at=a.recorded_at,
            transcript=a.transcript,
            transcript_status=a.transcript_status,
            response_time=a.response_time,
            speaking_duration=a.speaking_duration,
            silence_duration=a.silence_duration,
            word_count=a.word_count,
            words_per_minute=a.words_per_minute,
            pause_count=a.pause_count,
            answer_length=a.answer_length,
            relevance_score=a.relevance_score,
            concept_coverage=a.concept_coverage,
            detected_concepts=detected,
            missing_concepts=missing,
            concept_explanations=explanations,
            estimated_eye_contact_percentage=a.estimated_eye_contact_percentage,
            camera_facing_percentage=a.camera_facing_percentage,
            gaze_frames_total=a.gaze_frames_total,
            gaze_frames_analyzed=a.gaze_frames_analyzed,
            head_orientation_counts=a.head_orientation_counts,
            avg_yaw_deg=a.avg_yaw_deg,
            avg_pitch_deg=a.avg_pitch_deg,
        ))
    return resp


@app.post("/api/interview/answer", response_model=schemas.AnswerResponse, status_code=status.HTTP_201_CREATED)
def submit_answer(payload: schemas.AnswerSubmitRequest, db: Session = Depends(get_db)):
    """Save a recorded answer for a question in the current session."""
    session = db.query(models.InterviewSession).filter(models.InterviewSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="Session is already completed.")

    answer = models.InterviewAnswer(
        session_id=payload.session_id,
        question_id=payload.question_id,
        question_text=payload.question_text,
        audio_duration_seconds=payload.audio_duration_seconds,
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)

    return schemas.AnswerResponse(
        id=answer.id,
        session_id=answer.session_id,
        question_id=answer.question_id,
        question_text=answer.question_text,
        audio_duration_seconds=answer.audio_duration_seconds,
        recorded_at=answer.recorded_at,
    )


@app.post("/api/interview/session/{session_id}/complete")
def complete_interview_session(session_id: int, db: Session = Depends(get_db)):
    """Mark a session as completed and record the completion timestamp."""
    session = db.query(models.InterviewSession).filter(models.InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    session.status = "completed"
    session.completed_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "completed", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-TEXT & CONCEPT ANALYSIS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-speech", response_model=schemas.SpeechAnalysisResponse)
def analyze_speech_api(payload: schemas.SpeechAnalysisRequest):
    """
    Perform Speech-To-Text transcription and calculate measurable speech features:
    - transcript
    - response_time
    - speaking_duration
    - words_per_minute
    - pause_count
    - answer_length
    - silence_duration
    - number_of_words
    """
    result = analyze_answer(
        audio_b64=payload.audio_b64,
        response_time_seconds=payload.response_time_seconds,
        client_transcript=payload.client_transcript
    )

    return schemas.SpeechAnalysisResponse(
        status=result["status"],
        message=result["message"],
        transcript=result["transcript"],
        response_time=result["response_time"],
        speaking_duration=result["speaking_duration"],
        words_per_minute=result["words_per_minute"],
        pause_count=result["pause_count"],
        answer_length=str(result["answer_length"]),
        silence_duration=result.get("silence_duration", 0.0),
        number_of_words=result.get("number_of_words", 0)
    )

@app.post("/api/interview/transcribe", response_model=schemas.TranscribeResponse)
def transcribe_answer(payload: schemas.TranscribeRequest, db: Session = Depends(get_db)):

    """
    Accept a base64-encoded WAV recording for one interview answer.
    Runs speech-to-text, speech metrics calculation, and explainable answer concept analysis.
    Persists all results back to the InterviewAnswer DB row.
    """
    # Validate answer_id
    if payload.answer_id < 1:
        raise HTTPException(status_code=400, detail="answer_id must be a positive integer.")

    answer_row = db.query(models.InterviewAnswer).filter(
        models.InterviewAnswer.id == payload.answer_id
    ).first()
    if not answer_row:
        raise HTTPException(status_code=404, detail="Interview answer record not found.")

    # Validate audio payload
    if not payload.audio_b64 or len(payload.audio_b64) < 10:
        raise HTTPException(status_code=400, detail="audio_b64 is empty or too short.")

    # Run speech analysis pipeline
    result = analyze_answer(
        audio_b64=payload.audio_b64,
        response_time_seconds=payload.response_time_seconds,
        client_transcript=payload.client_transcript,
    )

    # Perform explainable concept analysis against question's expected concepts
    q_map = {q["id"]: q for q in QUESTIONS}
    question_obj = q_map.get(answer_row.question_id, {})
    expected_concepts = question_obj.get("expected_concepts", [])

    concept_res = analyze_answer_concepts(
        transcript=result["transcript"],
        expected_concepts=expected_concepts,
        question_text=answer_row.question_text or question_obj.get("text", "")
    )

    # Persist all metrics back to the DB row
    answer_row.transcript         = result["transcript"]
    answer_row.transcript_status  = result["status"]
    answer_row.response_time      = int(result["response_time"])
    answer_row.speaking_duration  = int(result["speaking_duration"])
    answer_row.silence_duration   = int(result["silence_duration"])
    answer_row.word_count         = result["word_count"]
    answer_row.words_per_minute   = int(result["words_per_minute"])
    answer_row.pause_count        = result["pause_count"]
    answer_row.answer_length      = len(result["transcript"])

    # Persist concept analysis
    answer_row.relevance_score      = concept_res["relevance_score"]
    answer_row.concept_coverage     = int(round(concept_res["concept_coverage"] * 100))
    answer_row.detected_concepts    = json.dumps(concept_res["detected_concepts"])
    answer_row.missing_concepts     = json.dumps(concept_res["missing_concepts"])
    answer_row.concept_explanations = json.dumps(concept_res["concept_explanations"])
    db.commit()

    concept_metrics = schemas.ConceptAnalysisMetrics(
        relevance_score      = concept_res["relevance_score"],
        concept_coverage     = concept_res["concept_coverage"],
        detected_concepts    = concept_res["detected_concepts"],
        missing_concepts     = concept_res["missing_concepts"],
        concept_explanations = concept_res["concept_explanations"],
        answer_length        = concept_res["answer_length"],
        word_count           = concept_res["word_count"],
        disclaimer           = concept_res["disclaimer"],
    )

    speech_metrics = schemas.SpeechMetrics(
        transcript        = result["transcript"],
        transcript_status = result["status"],
        message           = result["message"],
        response_time     = result["response_time"],
        speaking_duration = result["speaking_duration"],
        silence_duration  = result["silence_duration"],
        words_per_minute  = result["words_per_minute"],
        word_count        = result["word_count"],
        pause_count       = result["pause_count"],
        answer_length     = len(result["transcript"]),
        concept_analysis  = concept_metrics,
    )

    return schemas.TranscribeResponse(
        answer_id=payload.answer_id,
        metrics=speech_metrics,
        concept_analysis=concept_metrics
    )


@app.post("/api/interview/analyze-answer", response_model=schemas.ConceptAnalysisMetrics)
def analyze_answer_endpoint(payload: schemas.AnalyzeAnswerRequest, db: Session = Depends(get_db)):
    """
    Analyze student's transcript against expected concepts.
    Calculates concept coverage, answer relevance, missing concepts, answer length,
    and explainable match reasons.
    Stores/updates the DB record if answer_id is supplied.
    """
    transcript = payload.transcript
    expected_concepts = payload.expected_concepts
    question_text = payload.question_text

    answer_row = None
    if payload.answer_id and payload.answer_id > 0:
        answer_row = db.query(models.InterviewAnswer).filter(models.InterviewAnswer.id == payload.answer_id).first()
        if answer_row:
            if not transcript and answer_row.transcript:
                transcript = answer_row.transcript
            if not expected_concepts and answer_row.question_id:
                q_map = {q["id"]: q for q in QUESTIONS}
                q_obj = q_map.get(answer_row.question_id, {})
                expected_concepts = q_obj.get("expected_concepts", [])
            if not question_text:
                question_text = answer_row.question_text

    concept_res = analyze_answer_concepts(
        transcript=transcript,
        expected_concepts=expected_concepts,
        question_text=question_text
    )

    if answer_row:
        answer_row.relevance_score      = concept_res["relevance_score"]
        answer_row.concept_coverage     = int(round(concept_res["concept_coverage"] * 100))
        answer_row.detected_concepts    = json.dumps(concept_res["detected_concepts"])
        answer_row.missing_concepts     = json.dumps(concept_res["missing_concepts"])
        answer_row.concept_explanations = json.dumps(concept_res["concept_explanations"])
        if transcript and not answer_row.transcript:
            answer_row.transcript = transcript
            answer_row.answer_length = len(transcript)
            answer_row.word_count = concept_res["word_count"]
        db.commit()

    return schemas.ConceptAnalysisMetrics(
        relevance_score      = concept_res["relevance_score"],
        concept_coverage     = concept_res["concept_coverage"],
        detected_concepts    = concept_res["detected_concepts"],
        missing_concepts     = concept_res["missing_concepts"],
        concept_explanations = concept_res["concept_explanations"],
        answer_length        = concept_res["answer_length"],
        word_count           = concept_res["word_count"],
        disclaimer           = concept_res["disclaimer"],
    )



@app.post("/api/analyze-gaze", response_model=schemas.GazeMetrics)
def analyze_gaze_standalone(payload: schemas.StandaloneGazeRequest, db: Session = Depends(get_db)):
    """
    Standalone endpoint to analyze webcam frames captured during an answer for:
    - Estimated Eye Contact Percentage
    - Camera-Facing Percentage
    - Head Orientation (Center, Left, Right, Up, Down)
    """
    if not payload.frames_b64:
        raise HTTPException(status_code=400, detail="frames_b64 list is empty.")

    result = analyze_gaze_frames(payload.frames_b64)
    orientation_raw = result.get("head_orientation_counts", {})

    if payload.answer_id and payload.answer_id > 0:
        answer_row = db.query(models.InterviewAnswer).filter(
            models.InterviewAnswer.id == payload.answer_id
        ).first()
        if answer_row:
            answer_row.estimated_eye_contact_percentage = int(round(result["estimated_eye_contact_percentage"]))
            answer_row.camera_facing_percentage         = int(round(result["camera_facing_percentage"]))
            answer_row.gaze_frames_total                = result["total_frames"]
            answer_row.gaze_frames_analyzed             = result["analyzed_frames"]
            answer_row.head_orientation_counts          = json.dumps(orientation_raw)
            answer_row.avg_yaw_deg                      = int(round(result["avg_yaw_deg"]))
            answer_row.avg_pitch_deg                    = int(round(result["avg_pitch_deg"]))
            db.commit()

    return schemas.GazeMetrics(
        total_frames                      = result["total_frames"],
        analyzed_frames                   = result["analyzed_frames"],
        estimated_eye_contact_percentage  = result["estimated_eye_contact_percentage"],
        camera_facing_percentage          = result["camera_facing_percentage"],
        head_orientation_counts           = schemas.HeadOrientationCounts(
            center = orientation_raw.get("center", 0),
            left   = orientation_raw.get("left",   0),
            right  = orientation_raw.get("right",  0),
            up     = orientation_raw.get("up",     0),
            down   = orientation_raw.get("down",   0),
        ),
        avg_yaw_deg   = result["avg_yaw_deg"],
        avg_pitch_deg = result["avg_pitch_deg"],
        status        = result["status"],
        message       = result["message"],
    )


@app.post("/api/interview/gaze", response_model=schemas.GazeResponse)
def analyze_gaze(payload: schemas.GazeRequest, db: Session = Depends(get_db)):

    """
    Accept a list of base64-encoded webcam frames captured during one answer.
    Runs head-pose and iris-gaze analysis on every frame using MediaPipe FaceLandmarker.
    Persists aggregated results to the InterviewAnswer row, then returns the metrics.

    Measurements returned (no psychological claims):
        estimated_eye_contact_percentage  – geometric iris-to-eye-centre offset
        camera_facing_percentage          – head yaw/pitch within ±22°/±18° window
        head_orientation_counts           – frame counts per orientation label

    Error handling:
        answer_id not found        → 404
        empty frames list          → 400
        no face in any frame       → 200 with status="no_face"
        model / runtime failure    → 200 with status="error"
    """
    if payload.answer_id < 1:
        raise HTTPException(status_code=400, detail="answer_id must be a positive integer.")

    answer_row = db.query(models.InterviewAnswer).filter(
        models.InterviewAnswer.id == payload.answer_id
    ).first()
    if not answer_row:
        raise HTTPException(status_code=404, detail="Interview answer record not found.")

    if not payload.frames_b64:
        raise HTTPException(status_code=400, detail="frames_b64 list is empty.")

    # Run analysis (never raises — returns status="error" dict on failure)
    result = analyze_gaze_frames(payload.frames_b64)

    orientation_raw = result.get("head_orientation_counts", {})

    # Persist to DB
    answer_row.estimated_eye_contact_percentage = int(round(result["estimated_eye_contact_percentage"]))
    answer_row.camera_facing_percentage         = int(round(result["camera_facing_percentage"]))
    answer_row.gaze_frames_total                = result["total_frames"]
    answer_row.gaze_frames_analyzed             = result["analyzed_frames"]
    answer_row.head_orientation_counts          = json.dumps(orientation_raw)
    answer_row.avg_yaw_deg                      = int(round(result["avg_yaw_deg"]))
    answer_row.avg_pitch_deg                    = int(round(result["avg_pitch_deg"]))
    db.commit()

    metrics = schemas.GazeMetrics(
        total_frames                      = result["total_frames"],
        analyzed_frames                   = result["analyzed_frames"],
        estimated_eye_contact_percentage  = result["estimated_eye_contact_percentage"],
        camera_facing_percentage          = result["camera_facing_percentage"],
        head_orientation_counts           = schemas.HeadOrientationCounts(
            center = orientation_raw.get("center", 0),
            left   = orientation_raw.get("left",   0),
            right  = orientation_raw.get("right",  0),
            up     = orientation_raw.get("up",     0),
            down   = orientation_raw.get("down",   0),
        ),
        avg_yaw_deg   = result["avg_yaw_deg"],
        avg_pitch_deg = result["avg_pitch_deg"],
        status        = result["status"],
        message       = result["message"],
    )

    return schemas.GazeResponse(answer_id=payload.answer_id, metrics=metrics)


# ── ML Model Loading & Prediction Endpoint ───────────────────────────────────

ML_MODEL_PATH = os.path.join(_ROOT, "models", "model_pipeline.joblib")
if not os.path.exists(ML_MODEL_PATH):
    ML_MODEL_PATH = os.path.join(_ROOT, "ml", "models", "best_tuned_pipeline.joblib")
if not os.path.exists(ML_MODEL_PATH):
    ML_MODEL_PATH = os.path.join(_ROOT, "ml", "models", "best_pipeline.joblib")

_model_pipeline = None

def get_model_pipeline():
    global _model_pipeline
    if _model_pipeline is None and os.path.exists(ML_MODEL_PATH):
        try:
            ml_dir = os.path.join(_ROOT, "ml")
            if ml_dir not in sys.path:
                sys.path.insert(0, ml_dir)
            import preprocessing as ml_prep
            sys.modules["preprocessing"] = ml_prep
        except Exception:
            pass

        try:
            import joblib
            _model_pipeline = joblib.load(ML_MODEL_PATH)
            print(f"[startup] Successfully loaded ML pipeline from: {ML_MODEL_PATH}")
        except Exception as e:
            print(f"[startup] Error loading ML pipeline: {e}")
    return _model_pipeline



@app.post("/predict", response_model=schemas.PredictionResponse)
@app.post("/api/predict", response_model=schemas.PredictionResponse)
def predict_performance(payload: schemas.PredictionRequest):
    """
    Predict mock interview performance category based on feature metrics.

    Accepts interview feature vector:
      - estimated_eye_contact_percentage (float, 0–100)
      - camera_facing_percentage (float, 0–100)
      - response_time (float, seconds)
      - speaking_duration (float, seconds)
      - words_per_minute (float)
      - pause_count (int)
      - answer_length (int)
      - concept_coverage (float, 0.0–1.0)
      - question_difficulty (str: "easy" | "medium" | "hard")

    COMPLIANCE DISCLAIMER:
      - This prediction is for educational feedback in mock practice only.
      - It DOES NOT constitute a hiring or employment decision.
      - It DOES NOT measure or claim to measure candidate personality, character, or psychological state.
    """
    pipe = get_model_pipeline()
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="ML model pipeline is not available. Please ensure model training/tuning has been executed."
        )

    # Extract feature values from request payload (support nested 'features' or top-level)
    feats = payload.features or schemas.InterviewFeatureVector()

    eye_c = payload.estimated_eye_contact_percentage if payload.estimated_eye_contact_percentage is not None else feats.estimated_eye_contact_percentage
    cam_f = payload.camera_facing_percentage if payload.camera_facing_percentage is not None else feats.camera_facing_percentage
    resp_t = payload.response_time if payload.response_time is not None else feats.response_time
    spk_d = payload.speaking_duration if payload.speaking_duration is not None else feats.speaking_duration
    wpm = payload.words_per_minute if payload.words_per_minute is not None else feats.words_per_minute
    pauses = payload.pause_count if payload.pause_count is not None else feats.pause_count
    ans_len = payload.answer_length if payload.answer_length is not None else feats.answer_length
    cov = payload.concept_coverage if payload.concept_coverage is not None else feats.concept_coverage
    diff = payload.question_difficulty if payload.question_difficulty is not None else feats.question_difficulty

    # Normalize concept_coverage if passed as percentage (e.g. 70 -> 0.70)
    if cov is not None and cov > 1.0:
        cov = cov / 100.0

    feature_dict = {
        "estimated_eye_contact_percentage": float(eye_c if eye_c is not None else 70.0),
        "camera_facing_percentage": float(cam_f if cam_f is not None else 80.0),
        "response_time": float(resp_t if resp_t is not None else 3.0),
        "speaking_duration": float(spk_d if spk_d is not None else 25.0),
        "words_per_minute": float(wpm if wpm is not None else 120.0),
        "pause_count": int(pauses if pauses is not None else 1),
        "answer_length": int(ans_len if ans_len is not None else 200),
        "concept_coverage": float(cov if cov is not None else 0.70),
        "question_difficulty": str(diff if diff is not None else "medium"),
    }

    import pandas as pd
    input_df = pd.DataFrame([feature_dict])

    try:
        prediction_arr = pipe.predict(input_df)
        predicted_category = str(prediction_arr[0])

        probs_dict = {"low": 0.33, "medium": 0.34, "high": 0.33}
        confidence = 0.50

        if hasattr(pipe, "predict_proba"):
            raw_probs = pipe.predict_proba(input_df)[0]
            clf_classes = list(pipe.named_steps["clf"].classes_)
            for cls_name, prob in zip(clf_classes, raw_probs):
                if cls_name in probs_dict:
                    probs_dict[cls_name] = round(float(prob), 4)

            confidence = round(float(probs_dict.get(predicted_category, max(raw_probs))), 4)

        model_clf_name = pipe.named_steps["clf"].__class__.__name__

        return schemas.PredictionResponse(
            status="success",
            predicted_performance_category=predicted_category,
            confidence_score=confidence,
            probabilities=schemas.CategoryProbability(**probs_dict),
            model_name=model_clf_name,
            features_received=feature_dict,
            disclaimer=(
                "IMPORTANT NOTICE: This prediction represents an algorithmic evaluation of mock interview performance metrics "
                "for educational practice only. It DOES NOT constitute a hiring decision or employment recommendation, "
                "and DOES NOT measure or evaluate candidate personality, emotional intelligence, or psychological state."
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ── Unsupervised Analytics Endpoints (PCA + K-Means) ──────────────────────────

UNSUPERVISED_JSON_PATH = os.path.join(_ROOT, "ml", "models", "unsupervised_results.json")
UNSUPERVISED_MODEL_PATH = os.path.join(_ROOT, "models", "pca_kmeans_pipeline.joblib")
if not os.path.exists(UNSUPERVISED_MODEL_PATH):
    UNSUPERVISED_MODEL_PATH = os.path.join(_ROOT, "ml", "models", "pca_kmeans_pipeline.joblib")

_unsupervised_artifact = None

def get_unsupervised_artifact():
    global _unsupervised_artifact
    if _unsupervised_artifact is None and os.path.exists(UNSUPERVISED_MODEL_PATH):
        try:
            ml_dir = os.path.join(_ROOT, "ml")
            if ml_dir not in sys.path:
                sys.path.insert(0, ml_dir)
            import preprocessing as ml_prep
            sys.modules["preprocessing"] = ml_prep
        except Exception:
            pass

        try:
            import joblib
            _unsupervised_artifact = joblib.load(UNSUPERVISED_MODEL_PATH)
            print(f"[startup] Loaded unsupervised PCA+KMeans pipeline from: {UNSUPERVISED_MODEL_PATH}")
        except Exception as e:
            print(f"[startup] Error loading unsupervised pipeline: {e}")
    return _unsupervised_artifact


@app.get("/api/analytics/unsupervised", response_model=schemas.UnsupervisedAnalyticsResponse)
@app.get("/api/analytics/pca", response_model=schemas.UnsupervisedAnalyticsResponse)
@app.get("/api/analytics/clusters", response_model=schemas.UnsupervisedAnalyticsResponse)
def get_unsupervised_analytics():
    """
    Returns PCA 2D coordinates, explained variance, K-Means cluster assignments,
    and feature statistics across all mock interview sessions.

    Cluster Names:
      - Performance Pattern A
      - Performance Pattern B
      - Performance Pattern C
    """
    try:
        data = _load_unsupervised_results()
        return _analytics_response(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read unsupervised analytics data: {str(e)}"
        )


def _load_unsupervised_results():
    if not os.path.exists(UNSUPERVISED_JSON_PATH):
        try:
            from ml.unsupervised import run_unsupervised_pipeline
            return run_unsupervised_pipeline()
        except Exception as e:
            raise HTTPException(
                status_code=530,
                detail=f"Unsupervised results file not found and could not be generated: {e}"
            )

    with open(UNSUPERVISED_JSON_PATH, "r", encoding="utf-8") as result_file:
        data = json.load(result_file)

    # Older generated artifacts may predate the overall statistics fields. Derive
    # them from the stored, calculated sample features so the API remains useful
    # without requiring an offline training run.
    samples = data.get("pca_samples", [])
    dataset_info = data.setdefault("dataset_info", {})
    dataset_info.setdefault("total_rows", len(samples))
    dataset_info.setdefault("num_features", len(dataset_info.get("features", [])))
    dataset_info.setdefault("real_samples", sum(not sample.get("is_synthetic", False) for sample in samples))
    dataset_info.setdefault("synthetic_samples", sum(bool(sample.get("is_synthetic", False)) for sample in samples))

    if "overall_feature_statistics" not in data:
        feature_stats = {}
        feature_names = dataset_info.get("features", [])
        for feature_name in feature_names:
            values = [sample.get("features", {}).get(feature_name) for sample in samples]
            numeric_values = []
            for value in values:
                try:
                    if value is not None and not isinstance(value, bool):
                        numeric_values.append(float(value))
                except (TypeError, ValueError):
                    pass

            if numeric_values and len(numeric_values) == len([v for v in values if v is not None]):
                import statistics
                feature_stats[feature_name] = {
                    "mean": round(statistics.mean(numeric_values), 2),
                    "std": round(statistics.stdev(numeric_values), 2) if len(numeric_values) > 1 else 0.0,
                    "min": round(min(numeric_values), 2),
                    "max": round(max(numeric_values), 2),
                }
            else:
                counts = {}
                for value in values:
                    label = str(value) if value is not None else "missing"
                    counts[label] = counts.get(label, 0) + 1
                feature_stats[feature_name] = {"counts": counts}

        data["overall_feature_statistics"] = feature_stats
    return data


def _analytics_response(data):
    pca_info = data["pca_info"]
    return schemas.UnsupervisedAnalyticsResponse(
        status="success",
        explained_variance_ratio=pca_info["explained_variance_ratio"],
        total_explained_variance=pca_info["total_explained_variance"],
        cluster_names=data["cluster_names"],
        pca_samples=[schemas.PCASamplePoint(**sample) for sample in data["pca_samples"]],
        cluster_statistics=data["cluster_statistics"],
        dataset_info=data.get("dataset_info", {}),
        overall_feature_statistics=data.get("overall_feature_statistics", {}),
        disclaimer=data.get("disclaimer", schemas.UnsupervisedAnalyticsResponse.model_fields["disclaimer"].default),
    )


@app.get("/api/analytics/pca/coordinates", response_model=schemas.PCACoordinatesResponse)
def get_pca_coordinates():
    """Return calculated 2D PCA coordinates and each sample's neutral cluster for plotting."""
    try:
        data = _load_unsupervised_results()
        samples = data["pca_samples"]
        pca_info = data["pca_info"]
        return schemas.PCACoordinatesResponse(
            status="success",
            explained_variance_ratio=pca_info["explained_variance_ratio"],
            total_explained_variance=pca_info["total_explained_variance"],
            coordinates=[{
                "sample_id": point["sample_id"],
                "pca_x": point["pca_x"],
                "pca_y": point["pca_y"],
                "cluster_id": point["cluster_id"],
                "cluster_name": point["cluster_name"],
            } for point in samples],
            dataset_info=data.get("dataset_info", {}),
            disclaimer=data.get("disclaimer", schemas.UnsupervisedAnalyticsResponse.model_fields["disclaimer"].default),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load PCA coordinates: {e}")


@app.get("/api/analytics/clusters/assignments", response_model=schemas.ClusterAssignmentsResponse)
def get_cluster_assignments():
    """Return K-Means assignments and calculated feature statistics for every dataset sample."""
    try:
        data = _load_unsupervised_results()
        return schemas.ClusterAssignmentsResponse(
            status="success",
            cluster_names=data["cluster_names"],
            assignments=[{
                "sample_id": point["sample_id"],
                "cluster_id": point["cluster_id"],
                "cluster_name": point["cluster_name"],
                "is_synthetic": point.get("is_synthetic", False),
            } for point in data["pca_samples"]],
            cluster_statistics=data["cluster_statistics"],
            overall_feature_statistics=data.get("overall_feature_statistics", {}),
            dataset_info=data.get("dataset_info", {}),
            disclaimer=data.get("disclaimer", schemas.UnsupervisedAnalyticsResponse.model_fields["disclaimer"].default),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load cluster assignments: {e}")


@app.post("/api/analytics/cluster-predict", response_model=schemas.ClusterPredictResponse)
def predict_cluster_for_sample(payload: schemas.PredictionRequest):
    """
    Transform a live candidate feature vector into 2D PCA space and assign K-Means cluster pattern.
    """
    artifact = get_unsupervised_artifact()
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Unsupervised pipeline artifact is not loaded. Run ml/unsupervised.py first."
        )

    feats = payload.features or schemas.InterviewFeatureVector()

    eye_c = payload.estimated_eye_contact_percentage if payload.estimated_eye_contact_percentage is not None else feats.estimated_eye_contact_percentage
    cam_f = payload.camera_facing_percentage if payload.camera_facing_percentage is not None else feats.camera_facing_percentage
    resp_t = payload.response_time if payload.response_time is not None else feats.response_time
    spk_d = payload.speaking_duration if payload.speaking_duration is not None else feats.speaking_duration
    wpm = payload.words_per_minute if payload.words_per_minute is not None else feats.words_per_minute
    pauses = payload.pause_count if payload.pause_count is not None else feats.pause_count
    ans_len = payload.answer_length if payload.answer_length is not None else feats.answer_length
    cov = payload.concept_coverage if payload.concept_coverage is not None else feats.concept_coverage
    diff = payload.question_difficulty if payload.question_difficulty is not None else feats.question_difficulty

    if cov is not None and cov > 1.0:
        cov = cov / 100.0

    feature_dict = {
        "estimated_eye_contact_percentage": float(eye_c if eye_c is not None else 70.0),
        "camera_facing_percentage": float(cam_f if cam_f is not None else 80.0),
        "response_time": float(resp_t if resp_t is not None else 3.0),
        "speaking_duration": float(spk_d if spk_d is not None else 25.0),
        "words_per_minute": float(wpm if wpm is not None else 120.0),
        "pause_count": int(pauses if pauses is not None else 1),
        "answer_length": int(ans_len if ans_len is not None else 200),
        "concept_coverage": float(cov if cov is not None else 0.70),
        "question_difficulty": str(diff if diff is not None else "medium"),
    }

    import pandas as pd
    input_df = pd.DataFrame([feature_dict])

    try:
        preprocessor = artifact["preprocessor"]
        pca = artifact["pca"]
        kmeans = artifact["kmeans"]
        cnames = artifact["cluster_names"]

        X_prep = preprocessor.transform(input_df)
        X_pca = pca.transform(X_prep)
        cid = int(kmeans.predict(X_prep)[0])

        cname = cnames.get(cid, cnames.get(str(cid), f"Performance Pattern {cid}"))

        return schemas.ClusterPredictResponse(
            status="success",
            pca_x=round(float(X_pca[0, 0]), 4),
            pca_y=round(float(X_pca[0, 1]), 4),
            cluster_id=cid,
            cluster_name=cname,
            disclaimer=(
                "This cluster assignment identifies statistical similarity to historical mock interview performance patterns. "
                "It DOES NOT evaluate candidate psychology, personality, or hiring eligibility."
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cluster prediction failed: {str(e)}"
        )


# ── Final Interview Report Endpoint ───────────────────────────────────────────

@app.get("/api/interview/session/{session_id}/report", response_model=schemas.InterviewReportResponse)
def get_interview_session_report(session_id: int, db: Session = Depends(get_db)):
    """
    Generate comprehensive final interview report for a completed or in-progress session.
    Calculates:
      1. Candidate Profile & Interview Metadata
      2. Answer Performance (Relevance, Concept Coverage, Technical Score)
      3. Communication Metrics (Estimated Eye Contact, Camera-Facing, Speaking Rate, Response Time, Pause Count)
      4. Question-by-Question Analysis (Question, Student Transcript, Relevance, Detected Concepts, Missing Concepts)
      5. ML Analysis (Supervised Model Prediction, Unsupervised K-Means Cluster & PCA 2D Coordinates, Features Used)
    """
    session = db.query(models.InterviewSession).filter(models.InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    user = session.user
    if not user:
        raise HTTPException(status_code=404, detail="Candidate profile associated with session not found.")

    # Parse question IDs
    try:
        q_ids = json.loads(session.question_ids)
    except Exception:
        q_ids = []

    q_map = {q["id"]: q for q in QUESTIONS}
    session_questions = [q_map[qid] for qid in q_ids if qid in q_map]

    # Map answers by question_id
    answers_by_qid = {ans.question_id: ans for ans in session.answers}

    # Build question-by-question analysis list
    question_items = []
    relevance_scores = []
    coverage_scores = []
    eye_contact_scores = []
    camera_facing_scores = []
    wpm_scores = []
    response_times = []
    pause_counts = []
    speaking_durations = []
    answer_lengths = []

    for idx, q in enumerate(session_questions):
        ans = answers_by_qid.get(q["id"])

        if ans:
            # Parse detected / missing concepts and explanations
            try:
                det_concepts = json.loads(ans.detected_concepts) if ans.detected_concepts else []
            except Exception:
                det_concepts = []
            try:
                miss_concepts = json.loads(ans.missing_concepts) if ans.missing_concepts else []
            except Exception:
                miss_concepts = []
            try:
                explanations = json.loads(ans.concept_explanations) if ans.concept_explanations else {}
            except Exception:
                explanations = {}

            # Fallback if answer exists but concepts were not analyzed yet
            if not det_concepts and not miss_concepts and ans.transcript:
                concept_res = analyze_answer_concepts(
                    transcript=ans.transcript,
                    expected_concepts=q.get("expected_concepts", []),
                    question_text=q["text"]
                )
                det_concepts = concept_res["detected_concepts"]
                miss_concepts = concept_res["missing_concepts"]
                explanations = concept_res["concept_explanations"]
                ans.relevance_score = concept_res["relevance_score"]
                ans.concept_coverage = int(round(concept_res["concept_coverage"] * 100))
                ans.detected_concepts = json.dumps(det_concepts)
                ans.missing_concepts = json.dumps(miss_concepts)
                ans.concept_explanations = json.dumps(explanations)
                db.commit()

            rel_score = float(ans.relevance_score or 0.0)
            cov_score = float(ans.concept_coverage or 0.0)
            eye_contact = float(ans.estimated_eye_contact_percentage or 0.0)
            cam_facing = float(ans.camera_facing_percentage or 0.0)
            wpm = float(ans.words_per_minute or 0.0)
            resp_time = float(ans.response_time or 0.0)
            spk_dur = float(ans.speaking_duration or 0.0)
            pauses = int(ans.pause_count or 0)
            ans_len = int(ans.answer_length or len(ans.transcript or ""))
            transcript_text = ans.transcript or "(No speech transcribed)"

            relevance_scores.append(rel_score)
            coverage_scores.append(cov_score)
            eye_contact_scores.append(eye_contact)
            camera_facing_scores.append(cam_facing)
            wpm_scores.append(wpm)
            response_times.append(resp_time)
            pause_counts.append(pauses)
            speaking_durations.append(spk_dur)
            answer_lengths.append(ans_len)
        else:
            # Unanswered question placeholder with expected defaults
            rel_score = 0.0
            cov_score = 0.0
            eye_contact = 0.0
            cam_facing = 0.0
            wpm = 0.0
            resp_time = 0.0
            spk_dur = 0.0
            pauses = 0
            det_concepts = []
            miss_concepts = q.get("expected_concepts", [])
            explanations = {}
            transcript_text = "(Question not yet recorded)"

        question_items.append(schemas.QuestionReportItem(
            question_number=idx + 1,
            question_id=q["id"],
            question=q["text"],
            topic=q.get("topic", "General"),
            difficulty=q.get("difficulty", session.difficulty),
            question_type=q.get("question_type", "technical"),
            student_transcript=transcript_text,
            answer_relevance=round(rel_score, 1),
            concept_coverage=round(cov_score, 1),
            detected_concepts=det_concepts,
            missing_concepts=miss_concepts,
            concept_explanations=explanations,
            estimated_eye_contact_percentage=round(eye_contact, 1),
            camera_facing_percentage=round(cam_facing, 1),
            speaking_rate=round(wpm, 1),
            response_time=round(resp_time, 1),
            speaking_duration=round(spk_dur, 1),
            pause_count=pauses
        ))

    # Aggregated averages for answered questions
    n_answered = len(relevance_scores)
    avg_relevance = round(float(np.mean(relevance_scores)), 1) if n_answered > 0 else 0.0
    avg_coverage = round(float(np.mean(coverage_scores)), 1) if n_answered > 0 else 0.0
    technical_score = round((avg_relevance * 0.5) + (avg_coverage * 0.5), 1)

    avg_eye_contact = round(float(np.mean(eye_contact_scores)), 1) if n_answered > 0 else 0.0
    avg_camera_facing = round(float(np.mean(camera_facing_scores)), 1) if n_answered > 0 else 0.0
    avg_wpm = round(float(np.mean(wpm_scores)), 1) if n_answered > 0 else 0.0
    avg_response_time = round(float(np.mean(response_times)), 1) if n_answered > 0 else 0.0
    total_pauses = int(sum(pause_counts)) if n_answered > 0 else 0
    total_speaking_duration = round(float(sum(speaking_durations)), 1) if n_answered > 0 else 0.0
    avg_speaking_duration = round(float(np.mean(speaking_durations)), 1) if n_answered > 0 else 0.0
    avg_answer_length = int(round(float(np.mean(answer_lengths)))) if n_answered > 0 else 0

    # Format date
    created_dt = session.created_at or datetime.now(timezone.utc)
    interview_date_str = created_dt.strftime("%B %d, %Y • %I:%M %p UTC")
    completed_dt_str = session.completed_at.strftime("%B %d, %Y • %I:%M %p UTC") if session.completed_at else None

    # Construct feature dictionary for ML evaluation
    ml_feature_dict = {
        "estimated_eye_contact_percentage": float(avg_eye_contact),
        "camera_facing_percentage": float(avg_camera_facing),
        "response_time": float(avg_response_time),
        "speaking_duration": float(avg_speaking_duration),
        "words_per_minute": float(avg_wpm),
        "pause_count": int(total_pauses),
        "answer_length": int(avg_answer_length),
        "concept_coverage": float(avg_coverage / 100.0),
        "question_difficulty": str(session.difficulty),
    }

    import pandas as pd
    input_df = pd.DataFrame([ml_feature_dict])

    # 1. Supervised Model Prediction
    model_prediction = "medium"
    confidence = 0.50
    probs_dict = {"low": 0.33, "medium": 0.34, "high": 0.33}
    supervised_model_name = "RandomForestClassifier (Supervised)"

    pipe = get_model_pipeline()
    if pipe is not None:
        try:
            pred_arr = pipe.predict(input_df)
            model_prediction = str(pred_arr[0])
            if hasattr(pipe, "predict_proba"):
                raw_probs = pipe.predict_proba(input_df)[0]
                clf_classes = list(pipe.named_steps["clf"].classes_)
                for cls_name, prob in zip(clf_classes, raw_probs):
                    if cls_name in probs_dict:
                        probs_dict[cls_name] = round(float(prob), 4)
                confidence = round(float(probs_dict.get(model_prediction, max(raw_probs))), 4)
            supervised_model_name = f"{pipe.named_steps['clf'].__class__.__name__} (Supervised)"
        except Exception as e:
            print(f"[report] Supervised prediction error: {e}")

    # 2. Unsupervised Model Clustering & PCA Projection
    cluster_id = 0
    cluster_name = "Performance Pattern A"
    pca_coords = [0.0, 0.0]
    unsupervised_model_name = "PCA (2D) + K-Means (K=3, Unsupervised)"

    unsup_art = get_unsupervised_artifact()
    if unsup_art is not None:
        try:
            u_prep = unsup_art["preprocessor"]
            u_pca = unsup_art["pca"]
            u_km = unsup_art["kmeans"]
            u_cnames = unsup_art["cluster_names"]

            X_prep = u_prep.transform(input_df)
            X_pca = u_pca.transform(X_prep)
            cid = int(u_km.predict(X_prep)[0])
            cname = u_cnames.get(cid, u_cnames.get(str(cid), f"Performance Pattern {cid}"))

            cluster_id = cid
            cluster_name = cname
            pca_coords = [round(float(X_pca[0, 0]), 4), round(float(X_pca[0, 1]), 4)]
        except Exception as e:
            print(f"[report] Unsupervised projection error: {e}")

    return schemas.InterviewReportResponse(
        status="success",
        candidate=schemas.CandidateReportInfo(
            full_name=user.full_name,
            student_id=user.student_id,
            email=user.email,
            target_role=user.target_role,
            experience_level=user.experience_level,
        ),
        interview=schemas.InterviewMetaInfo(
            session_id=session.id,
            role=session.role,
            interview_type=session.interview_type,
            difficulty=session.difficulty,
            interview_date=interview_date_str,
            completed_at=completed_dt_str,
            total_questions=len(session_questions),
            completed_questions=n_answered,
        ),
        answer_performance=schemas.AnswerPerformanceMetrics(
            answer_relevance=avg_relevance,
            concept_coverage=avg_coverage,
            technical_score=technical_score,
        ),
        communication_metrics=schemas.CommunicationReportMetrics(
            estimated_eye_contact_percentage=avg_eye_contact,
            camera_facing_percentage=avg_camera_facing,
            speaking_rate=avg_wpm,
            response_time=avg_response_time,
            pause_count=total_pauses,
            speaking_duration_total=total_speaking_duration,
        ),
        question_analysis=question_items,
        ml_analysis=schemas.MLAnalysisReport(
            model_prediction=model_prediction,
            confidence_score=confidence,
            probabilities=probs_dict,
            cluster_assignment=cluster_name,
            cluster_id=cluster_id,
            pca_coordinates=pca_coords,
            feature_values=ml_feature_dict,
            supervised_model_used=supervised_model_name,
            unsupervised_model_used=unsupervised_model_name,
            disclaimer=(
                "Compliance Notice: Machine learning models evaluate historical metric trends for skill development only. "
                "These classifications DO NOT measure psychological condition, personality, character, or hiring suitability."
            ),
        ),
        disclaimer=(
            "IMPORTANT NOTICE: This mock interview report is generated automatically using speech, gaze, and machine learning "
            "algorithms strictly for educational practice and candidate skill improvement. "
            "It DOES NOT constitute a psychological assessment, emotional evaluation, personality profile, or hiring/employment decision."
        ),
    )


# ── Interview History Endpoints ───────────────────────────────────────────────

@app.get("/api/interview/history", response_model=schemas.InterviewHistoryResponse)
@app.get("/api/history", response_model=schemas.InterviewHistoryResponse)
@app.get("/api/interview/history/{user_id}", response_model=schemas.InterviewHistoryResponse)
def get_interview_history(
    user_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Fetch complete interview history for a candidate or all recorded sessions.
    Stores and returns:
      - Session metadata (User, Date, Role, Interview Type, Difficulty, Status)
      - Complete Answers, Transcripts, Measurements, and Concept Analytics
      - Machine Learning Classifications & Patterns
      - Individual User Progress Trends over time (No competitive leaderboards).
    """
    if user_id is None or user_id < 1:
        raise HTTPException(status_code=400, detail="user_id is required to view private interview history.")

    query = db.query(models.InterviewSession)
    query = query.filter(models.InterviewSession.user_id == user_id)

    sessions = query.order_by(models.InterviewSession.created_at.desc()).limit(limit).all()

    pipe = get_model_pipeline()
    unsup_art = get_unsupervised_artifact()

    history_cards: List[schemas.InterviewHistoryCardItem] = []

    for sess in sessions:
        user = sess.user
        cand_name = user.full_name if user else "Candidate"
        stud_id = user.student_id if user else "N/A"
        t_role = user.target_role if user else sess.role

        try:
            q_ids = json.loads(sess.question_ids)
        except Exception:
            q_ids = []
        tot_q = max(len(q_ids), len(sess.answers))

        answers = sess.answers or []
        n_ans = len(answers)

        rel_scores = [float(a.relevance_score or 0) for a in answers]
        cov_scores = [float(a.concept_coverage or 0) for a in answers]
        eye_scores = [float(a.estimated_eye_contact_percentage or 0) for a in answers]
        cam_scores = [float(a.camera_facing_percentage or 0) for a in answers]
        wpm_scores = [float(a.words_per_minute or 0) for a in answers]
        resp_scores = [float(a.response_time or 0) for a in answers]
        pause_tot = sum([int(a.pause_count or 0) for a in answers])
        speak_tot = sum([float(a.speaking_duration or 0) for a in answers])

        avg_rel = round(float(np.mean(rel_scores)), 1) if n_ans > 0 else 0.0
        avg_cov = round(float(np.mean(cov_scores)), 1) if n_ans > 0 else 0.0
        tech_score = round((avg_rel * 0.5) + (avg_cov * 0.5), 1)
        avg_eye = round(float(np.mean(eye_scores)), 1) if n_ans > 0 else 0.0
        avg_cam = round(float(np.mean(cam_scores)), 1) if n_ans > 0 else 0.0
        avg_wpm = round(float(np.mean(wpm_scores)), 1) if n_ans > 0 else 0.0
        avg_resp = round(float(np.mean(resp_scores)), 1) if n_ans > 0 else 0.0

        # Run ML model prediction
        ml_pred = "medium"
        ml_cluster = "Performance Pattern A"

        feat_dict = {
            "estimated_eye_contact_percentage": float(avg_eye),
            "camera_facing_percentage": float(avg_cam),
            "response_time": float(avg_resp),
            "speaking_duration": float(speak_tot / max(1, n_ans)),
            "words_per_minute": float(avg_wpm),
            "pause_count": int(pause_tot),
            "answer_length": int(np.mean([a.answer_length or len(a.transcript or '') for a in answers])) if n_ans > 0 else 0,
            "concept_coverage": float(avg_cov / 100.0),
            "question_difficulty": str(sess.difficulty),
        }

        import pandas as pd
        input_df = pd.DataFrame([feat_dict])

        if pipe is not None:
            try:
                ml_pred = str(pipe.predict(input_df)[0])
            except Exception:
                pass

        if unsup_art is not None:
            try:
                u_prep = unsup_art["preprocessor"]
                u_km = unsup_art["kmeans"]
                u_cnames = unsup_art["cluster_names"]
                X_prep = u_prep.transform(input_df)
                cid = int(u_km.predict(X_prep)[0])
                ml_cluster = u_cnames.get(cid, u_cnames.get(str(cid), f"Performance Pattern {cid}"))
            except Exception:
                pass

        created_dt = sess.created_at or datetime.now(timezone.utc)
        date_str = created_dt.strftime("%b %d, %Y • %I:%M %p")
        created_iso = created_dt.isoformat()
        comp_iso = sess.completed_at.isoformat() if sess.completed_at else None

        card_metrics = schemas.HistoryCardMetrics(
            technical_score=tech_score,
            answer_relevance=avg_rel,
            concept_coverage=avg_cov,
            estimated_eye_contact=avg_eye,
            camera_facing=avg_cam,
            speaking_rate=avg_wpm,
            response_time=avg_resp,
            pause_count=pause_tot,
            speaking_duration=round(speak_tot, 1),
            ml_prediction=ml_pred,
            ml_cluster=ml_cluster,
        )

        history_cards.append(schemas.InterviewHistoryCardItem(
            session_id=sess.id,
            user_id=sess.user_id,
            candidate_name=cand_name,
            student_id=stud_id,
            target_role=t_role,
            interview_role=sess.role,
            interview_type=sess.interview_type,
            difficulty=sess.difficulty,
            status=sess.status or "completed",
            date_formatted=date_str,
            created_at_iso=created_iso,
            completed_at_iso=comp_iso,
            total_questions=tot_q,
            completed_questions=n_ans,
            metrics=card_metrics,
        ))

    # Chronological progression for individual user progress chart (oldest to newest)
    chronological = sorted(history_cards, key=lambda c: c.created_at_iso)
    progress_trend: List[schemas.UserProgressPoint] = []

    for idx, item in enumerate(chronological):
        progress_trend.append(schemas.UserProgressPoint(
            session_id=item.session_id,
            session_number=idx + 1,
            date_label=item.date_formatted.split("•")[0].strip(),
            created_at_iso=item.created_at_iso,
            role=item.interview_role,
            technical_score=item.metrics.technical_score,
            answer_relevance=item.metrics.answer_relevance,
            concept_coverage=item.metrics.concept_coverage,
            estimated_eye_contact=item.metrics.estimated_eye_contact,
            speaking_rate=item.metrics.speaking_rate,
            response_time=item.metrics.response_time,
        ))

    # Calculate overall progress summary
    total_int = len(history_cards)
    if total_int > 0:
        avg_tech = round(float(np.mean([c.metrics.technical_score for c in history_cards])), 1)
        avg_r = round(float(np.mean([c.metrics.answer_relevance for c in history_cards])), 1)
        avg_c = round(float(np.mean([c.metrics.concept_coverage for c in history_cards])), 1)
        avg_e = round(float(np.mean([c.metrics.estimated_eye_contact for c in history_cards])), 1)
        avg_w = round(float(np.mean([c.metrics.speaking_rate for c in history_cards])), 1)

        if len(chronological) >= 2:
            score_diff = round(chronological[-1].metrics.technical_score - chronological[0].metrics.technical_score, 1)
        else:
            score_diff = 0.0

        if score_diff > 0:
            trend_str = "improving"
        elif score_diff == 0:
            trend_str = "steady"
        else:
            trend_str = "needs_practice"
    else:
        avg_tech = 0.0
        avg_r = 0.0
        avg_c = 0.0
        avg_eye = 0.0
        avg_w = 0.0
        score_diff = 0.0
        trend_str = "steady"

    summary_obj = schemas.UserProgressSummary(
        total_interviews=total_int,
        avg_technical_score=avg_tech,
        avg_relevance=avg_r,
        avg_concept_coverage=avg_c,
        avg_eye_contact=avg_eye,
        avg_speaking_rate=avg_w,
        score_trend=trend_str,
        score_change=score_diff,
    )

    return schemas.InterviewHistoryResponse(
        status="success",
        interviews=history_cards,
        progress_trend=progress_trend,
        summary=summary_obj,
        disclaimer=(
            "Interview history and progress metrics track individual self-practice progression over time. "
            "These metrics are private to the candidate and are NOT used to rank, compare, or evaluate students against each other."
        ),
    )



