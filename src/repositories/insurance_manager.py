import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, JSON, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload

load_dotenv()

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    gender = Column(String(50))
    policies = relationship("Policy", back_populates="customer")

class Policy(Base):
    __tablename__ = 'policies'
    policy_id = Column(String(50), primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    plan_name = Column(String(100))
    status = Column(String(50))
    customer = relationship("Customer", back_populates="policies")

class Claim(Base):
    __tablename__ = 'claims'
    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_code = Column(String(50), unique=True, nullable=False)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'))
    claim_type = Column(Enum('medical', 'accident'), nullable=False)
    status = Column(String(50), default='pending_review')
    docs_required = Column(JSON)
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class MedicalDetail(Base):
    __tablename__ = 'medical_claim_details'
    claim_id = Column(Integer, ForeignKey('claims.id'), primary_key=True)
    hospital_name = Column(String(255))
    admission_date = Column(String(50))
    diagnosis = Column(Text)

class AccidentDetail(Base):
    __tablename__ = 'accident_claim_details'
    claim_id = Column(Integer, ForeignKey('claims.id'), primary_key=True)
    incident_date = Column(String(50))
    incident_description = Column(Text)
    injury_type = Column(String(255))

class InsuranceManager:
    def __init__(self):
        user = os.getenv("DB_USER")
        pw = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", 3306)
        db = os.getenv("DB_NAME")
        
        db_url = f"mysql+mysqlconnector://{user}:{pw}@{host}:{port}/{db}"
        
        try:
            self.engine = create_engine(db_url, echo=False, pool_pre_ping=True)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            print("Successfully connected to the database via ORM.")
        except Exception as e:
            print(f"Error connecting to Database: {e}")
            self.session = None

    def verify_policy(self, customer_name, policy_id):
        if not self.session: return None

        result = self.session.query(Policy).join(Customer).filter(
            Policy.policy_id == policy_id,
            Customer.full_name == customer_name
        ).first()

        if result:
            return {
                "holder": result.customer.full_name,
                "plan": result.plan_name,
                "status": result.status
            }
        return {"status": "not_found"}

    def process_medical_claim(self, policy_id, hospital_name, admission_date, diagnosis):
        if not self.session: return None
        
        claim_code = f"CLM-{uuid.uuid4().hex[:6].upper()}"
        docs = ["VAT Invoice", "Discharge Summary"]
        
        try:
            new_claim = Claim(
                claim_code=claim_code,
                policy_id=policy_id,
                claim_type='medical',
                status='pending_review',
                docs_required=docs,
                note="Processing medical claim via ORM"
            )
            self.session.add(new_claim)
            self.session.flush()

            detail = MedicalDetail(
                claim_id=new_claim.id,
                hospital_name=hospital_name,
                admission_date=admission_date,
                diagnosis=diagnosis
            )
            self.session.add(detail)
            
            self.session.commit()
            return {
                "claim_id": claim_code, 
                "status": "pending_review", 
                "docs_required": docs, 
                "note": new_claim.note
            }
        except Exception as e:
            self.session.rollback()
            print(f"Error processing medical claim: {e}")
            return None

    def process_accident_claim(self, policy_id, incident_date, incident_description, injury_type):
        if not self.session: return None

        claim_code = f"ACC-{uuid.uuid4().hex[:6].upper()}"
        docs = ["Police Report", "Medical Evidence"]

        try:
            new_claim = Claim(
                claim_code=claim_code,
                policy_id=policy_id,
                claim_type='accident',
                status='pending_review',
                docs_required=docs,
                note="Processing accident claim via ORM"
            )
            self.session.add(new_claim)
            self.session.flush()

            detail = AccidentDetail(
                claim_id=new_claim.id,
                incident_date=incident_date,
                incident_description=incident_description,
                injury_type=injury_type
            )
            self.session.add(detail)

            self.session.commit()
            return {
                "claim_id": claim_code, 
                "status": "pending_review", 
                "docs_required": docs, 
                "note": new_claim.note
            }
        except Exception as e:
            self.session.rollback()
            print(f"Error processing accident claim: {e}")
            return None

    def close(self):
        """Đóng session"""
        if self.session:
            self.session.close()
            print("Database session closed.")