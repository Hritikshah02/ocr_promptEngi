from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import date

class DrivingLicense(BaseModel):
    name: str
    date_of_birth: Optional[date] = None
    license_number: str
    issuing_state: Optional[str] = ""
    expiry_date: Optional[date] = None

class ReceiptItem(BaseModel):
    name: str
    quantity: int
    price: float

class ShopReceipt(BaseModel):
    merchant_name: str
    total_amount: float
    date_of_purchase: Optional[date] = None
    items: List[ReceiptItem] = []
    payment_method: Optional[str] = ""

class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None

class Education(BaseModel):
    institution: str
    degree: str
    graduation_year: Optional[int] = None

class Resume(BaseModel):
    full_name: str
    email: EmailStr = "unknown@example.com"  # Default value that passes EmailStr validation
    phone_number: str = ""
    skills: List[str] = []
    work_experience: List[WorkExperience] = []
    education: List[Education] = []
