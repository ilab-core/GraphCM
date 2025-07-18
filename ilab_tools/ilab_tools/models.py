
from sqlalchemy import Column, Enum, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Company(Base):
    __tablename__ = "company"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    domain = Column(String(128))
    status = Column(Enum('0', '1'))
