from sqlalchemy import and_

from .models import Company


def status_save(date, company_name, column_name, new_status, session, table_name):
    company_entry = session.query(Company).filter(
        Company.domain == company_name).one_or_none()
    if company_entry:
        status_entry = session.query(table_name).filter(
            and_(table_name.tarih == date, table_name.company_id == company_entry.id)).one_or_none()
        if status_entry:
            setattr(status_entry, column_name.name, new_status)
        else:
            status_entry = table_name(tarih=date, company_id=company_entry.id)
            setattr(status_entry, column_name.name, new_status)
            session.add(status_entry)
        session.commit()
