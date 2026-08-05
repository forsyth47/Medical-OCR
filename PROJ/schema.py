from typing import List, Optional
from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str = Field(description="Medicine name (generic or brand)")
    dose: Optional[str] = Field(None, description="e.g., 40 mg, 25 mcg")
    timing: Optional[str] = Field(None, description="e.g., 1-0-1, once daily, weekly")
    duration: Optional[str] = Field(None, description="e.g., x 15 days, x 8 weeks")
    note: Optional[str] = None


class Prescription(BaseModel):
    clinic: Optional[str] = None
    date: Optional[str] = None
    patient_name: Optional[str] = None
    diagnoses: Optional[str] = None
    medications: List[Medication] = Field(default_factory=list)
    raw_transcription: Optional[str] = None