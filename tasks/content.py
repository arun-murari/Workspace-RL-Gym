EMAIL_CONTENT = [
    {"key": "budget",
     "subject": "Q3 Budget Review",
     "body": ("Hi, please review the Q3 budget before we finalize. Total projected spend is $327K, "
              "with engineering at $180K and marketing at $90K. Headcount increases by 3. "
              "Please confirm approval by March 14th."),
     "facts": {"total": "$327K", "engineering": "$180K", "deadline": "March 14th"}},

    {"key": "invoice",
     "subject": "Invoice 4471 from Acme Corp",
     "body": ("Attached is invoice #4471 from Acme Corp. The amount due is $12,400, "
              "payable under Net 30 terms. Payment is due by April 2nd. PO number is 318."),
     "facts": {"invoice_no": "4471", "amount": "$12,400", "deadline": "April 2nd", "po": "318"}},

    {"key": "project",
     "subject": "Project Atlas Status",
     "body": ("Team, Project Atlas is now 75% complete. The launch has moved to May 9th. "
              "The remaining budget is $42K. Carol is leading QA and David owns deployment."),
     "facts": {"completion": "75%", "launch": "May 9th", "remaining_budget": "$42K"}},

    {"key": "meeting",
     "subject": "Notes: Security Audit Review",
     "body": ("Notes from the security audit meeting. We found 7 open issues, 2 of them critical. "
              "Remediation is targeted for June 1st. The auditor was CloudSoft. Next review is in Q3."),
     "facts": {"open_issues": "7", "critical": "2", "remediation": "June 1st", "auditor": "CloudSoft"}},

    {"key": "report",
     "subject": "March Finance Report",
     "body": ("Attached is the March finance report. Revenue was $540K against expenses of $310K, "
              "leaving a net of $230K. Open AR is $88K. Status is green for the quarter."),
     "facts": {"revenue": "$540K", "expenses": "$310K", "net": "$230K", "status": "green"}},

    {"key": "contract",
     "subject": "Contract Renewal — TechSupply",
     "body": ("The TechSupply contract is up for renewal. The new annual value is $96K over a 2 year term. "
              "Legal has approved the terms. We need a signature by February 28th."),
     "facts": {"value": "$96K", "term": "2 year", "deadline": "February 28th"}},

    {"key": "hiring",
     "subject": "Hiring Plan Q3",
     "body": ("The Q3 hiring plan adds 5 engineers and 2 designers. The total comp budget is $1.2M. "
              "First start date is targeted for July 7th. Recruiting is led by Eve."),
     "facts": {"engineers": "5", "designers": "2", "comp_budget": "$1.2M", "start": "July 7th"}},

    {"key": "launch",
     "subject": "Product Launch — Beacon",
     "body": ("Beacon launches on August 15th. Marketing spend is $75K and we expect 10K signups "
              "in the first month. The press embargo lifts at 9am ET on launch day."),
     "facts": {"launch_date": "August 15th", "marketing_spend": "$75K", "expected_signups": "10K"}},

    {"key": "roadmap",
     "subject": "2026 Roadmap Draft",
     "body": ("Here is the 2026 roadmap draft. We have 4 major milestones, the first due March 31st. "
              "The total planned investment is $3.4M across 3 product lines."),
     "facts": {"milestones": "4", "first_due": "March 31st", "investment": "$3.4M"}},

    {"key": "vendor",
     "subject": "Vendor Review — DataPro",
     "body": ("DataPro's annual review is complete. Their SLA uptime was 99.4%, just under the 99.5% target. "
              "Annual cost was $54K. Recommendation is to renew with a credit for the missed SLA."),
     "facts": {"uptime": "99.4%", "target": "99.5%", "cost": "$54K"}},

    {"key": "expense",
     "subject": "Expense Report Approval",
     "body": ("Your expense report for the conference is ready for approval. The total is $2,340, "
              "including $1,200 for travel and $640 for lodging. Submit by May 5th for reimbursement."),
     "facts": {"total": "$2,340", "travel": "$1,200", "deadline": "May 5th"}},

    {"key": "outage",
     "subject": "Incident Report — API Outage",
     "body": ("The API outage on Tuesday lasted 47 minutes and affected 3 services. Root cause was a "
              "database failover. A postmortem is scheduled for April 18th. Owner is Frank."),
     "facts": {"duration": "47 minutes", "services": "3", "postmortem": "April 18th"}},

    {"key": "renewal",
     "subject": "License Renewal — CloudSoft",
     "body": ("Our CloudSoft license renews on September 1st. The annual cost rose to $38K, up 8% from last year. "
              "We have 120 seats. Finance needs sign-off by August 20th."),
     "facts": {"renewal_date": "September 1st", "cost": "$38K", "seats": "120"}},

    {"key": "survey",
     "subject": "Q2 Employee Survey Results",
     "body": ("The Q2 employee survey is in. Overall satisfaction was 82%, up from 78% last quarter. "
              "Response rate was 91%. The top concern raised was workload balance."),
     "facts": {"satisfaction": "82%", "response_rate": "91%", "top_concern": "workload balance"}},
]

# This is a curated list of possible email content that the tasks can draw from

FILE_CONTENT = [
    {"name": "q3_budget.xlsx",
     "content": ("FY2026 Q3 Budget. Total allocated: $327,000. Engineering: $180K, Marketing: $90K, "
                 "Operations: $57K. Approved by finance on January 15th."),
     "facts": {"total": "$327,000", "engineering": "$180K", "approved": "January 15th"}},

    {"name": "atlas_plan.docx",
     "content": ("Project Atlas Plan. Objective: ship the v2 platform by May 9th. Budget: $250K. "
                 "Team of 8. Primary risk is the data migration in phase 2."),
     "facts": {"deadline": "May 9th", "budget": "$250K", "team_size": "8"}},

    {"name": "acme_contract.pdf",
     "content": ("Service Contract with Acme Corp. Total value: $12,400. Term: 1 year. "
                 "Auto-renews unless cancelled 30 days prior. Signed January 8th."),
     "facts": {"value": "$12,400", "term": "1 year", "signed": "January 8th"}},

    {"name": "march_report.pdf",
     "content": ("March Department Report. Revenue: $540K. Expenses: $310K. Net: $230K. "
                 "Headcount: 42. Open issues: 5. Overall status: Green."),
     "facts": {"revenue": "$540K", "net": "$230K", "headcount": "42"}},

    {"name": "security_audit.md",
     "content": ("# Security Audit Findings\n\nTotal issues: 7. Critical: 2. High: 3. Low: 2.\n"
                 "Auditor: CloudSoft. Remediation deadline: June 1st."),
     "facts": {"total_issues": "7", "critical": "2", "auditor": "CloudSoft"}},

    {"name": "hiring_plan.xlsx",
     "content": ("Q3 Hiring Plan. Engineers: 5. Designers: 2. PMs: 1. Total comp budget: $1.2M. "
                 "First start date: July 7th."),
     "facts": {"engineers": "5", "comp_budget": "$1.2M", "start": "July 7th"}},

    {"name": "beacon_launch.pptx",
     "content": ("Beacon Launch Deck. Launch date: August 15th. Marketing spend: $75K. "
                 "Target: 10K signups month one. Channels: email, paid social, PR."),
     "facts": {"launch_date": "August 15th", "marketing_spend": "$75K"}},

    {"name": "roadmap_2026.md",
     "content": ("# 2026 Roadmap\n\n4 milestones. First milestone due March 31st. "
                 "Total investment: $3.4M. Three product lines: Atlas, Beacon, Cosmos."),
     "facts": {"milestones": "4", "first_due": "March 31st", "investment": "$3.4M"}},

    {"name": "vendor_review.docx",
     "content": ("DataPro Vendor Review. SLA uptime: 99.4% against 99.5% target. Annual cost: $54K. "
                 "Recommendation: renew with SLA credit. Reviewer: Grace."),
     "facts": {"uptime": "99.4%", "cost": "$54K"}},

    {"name": "incident_postmortem.md",
     "content": ("# API Outage Postmortem\n\nDuration: 47 minutes. Services affected: 3. "
                 "Root cause: database failover. Action items: 4. Owner: Frank."),
     "facts": {"duration": "47 minutes", "services": "3", "action_items": "4"}},

    {"name": "license_summary.xlsx",
     "content": ("CloudSoft License Summary. Renewal: September 1st. Annual cost: $38K. Seats: 120. "
                 "Utilization: 87%. Renewal owner: Finance."),
     "facts": {"renewal": "September 1st", "cost": "$38K", "seats": "120"}},
]

# This is the possible contents for a file.

# I decided that building up examples/templates for how each email and file would look allows for richer tasks and simulates and more realistic
# environment. Allowing for tasks like summary and making sure that despite