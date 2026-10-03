"""Script to generate realistic test PDF documents for demonstration and evaluation."""
from pathlib import Path
import pymupdf


def create_policy_2025_pdf(dest_path: Path):
    """Create 2025 version of company handbook."""
    doc = pymupdf.open()

    # Page 1: Overview
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Northstar Analytics — Employee Operations Handbook (2025 Edition)", fontsize=16)
    p1.insert_text(
        (50, 110),
        "1. Company Overview\n"
        "Northstar Analytics Pvt. Ltd. is a business-intelligence software company headquartered in Bengaluru.\n"
        "Founded in 2019, the company builds analytics and automated reporting dashboards for mid-sized enterprises.\n"
        "The engineering organization utilizes Python for backend services and PostgreSQL for data persistence.\n\n"
        "Document Version: Effective January 1, 2025 to December 31, 2025.",
        fontsize=11,
    )

    # Page 2: Leave & Attendance
    p2 = doc.new_page()
    p2.insert_text((50, 72), "2. Leave and Attendance Policy (2025)", fontsize=14)
    p2.insert_text(
        (50, 110),
        "Employees receive 15 days of paid annual leave during each calendar year.\n"
        "Employees also receive 10 days of paid sick leave per calendar year.\n"
        "A maximum of 5 unused annual leave days may be carried into the following year.\n"
        "Planned annual leave exceeding 3 working days must be requested at least 7 days in advance.",
        fontsize=11,
    )

    # Page 3: Remote Work & Notice Period
    p3 = doc.new_page()
    p3.insert_text((50, 72), "3. Remote Work and Separation Terms (2025)", fontsize=14)
    p3.insert_text(
        (50, 110),
        "Remote Work Guidelines:\n"
        "Eligible employees may work remotely for up to 2 working days per week with manager approval.\n"
        "Wednesday is the mandatory collaboration day in the office.\n\n"
        "Separation and Notice Period:\n"
        "Standard employee notice period upon resignation is 30 calendar days.\n"
        "Company property must be returned on or before the final working day.",
        fontsize=11,
    )

    doc.save(dest_path)
    doc.close()
    print(f"Created: {dest_path}")


def create_policy_2026_pdf(dest_path: Path):
    """Create 2026 version of company handbook with updated policies."""
    doc = pymupdf.open()

    # Page 1: Overview
    p1 = doc.new_page()
    p1.insert_text((50, 72), "Northstar Analytics — Employee Operations Handbook (2026 Edition)", fontsize=16)
    p1.insert_text(
        (50, 110),
        "1. Company Overview\n"
        "Northstar Analytics Pvt. Ltd. is a business-intelligence software company headquartered in Bengaluru.\n"
        "The company currently employs approximately 180 team members. Primary product is Northstar Pulse.\n"
        "The engineering stack uses Python, TypeScript, Docker, and Qdrant for vector search capabilities.\n\n"
        "Document Version: Effective January 1, 2026 and supersedes all previous editions.",
        fontsize=11,
    )

    # Page 2: Leave & Attendance
    p2 = doc.new_page()
    p2.insert_text((50, 72), "2. Leave and Attendance Policy (2026)", fontsize=14)
    p2.insert_text(
        (50, 110),
        "Employees receive 24 days of paid annual leave during each calendar year.\n"
        "Employees also receive 12 days of paid sick leave per calendar year.\n"
        "A maximum of 10 unused annual leave days may be carried into the following year.\n"
        "Planned annual leave exceeding 3 working days must be requested at least 7 days in advance.",
        fontsize=11,
    )

    # Page 3: Remote Work & Notice Period
    p3 = doc.new_page()
    p3.insert_text((50, 72), "3. Remote Work and Separation Terms (2026)", fontsize=14)
    p3.insert_text(
        (50, 110),
        "Remote Work Guidelines:\n"
        "Eligible employees may work remotely for up to 3 working days per week.\n"
        "Tuesday and Thursday are designated collaboration days in the office.\n\n"
        "Separation and Notice Period:\n"
        "Standard employee notice period upon resignation is 60 calendar days.\n"
        "During notice period, remote work is subject to HR director approval.",
        fontsize=11,
    )

    doc.save(dest_path)
    doc.close()
    print(f"Created: {dest_path}")


def main():
    dest_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)

    create_policy_2025_pdf(dest_dir / "policy_2025.pdf")
    create_policy_2026_pdf(dest_dir / "policy_2026.pdf")
    print("Sample test PDFs successfully created!")


if __name__ == "__main__":
    main()
