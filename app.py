import os
import gradio as gr
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# CONFIG
# ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please add it in HuggingFace Space Secrets.")

client = Groq(api_key=GROQ_API_KEY)
INDEX_PATH = "faiss_index"

# ==============================
# KNOWLEDGE BASE
# 4 Universities: COMSATS, NUST, UET Lahore, QAU
# ==============================
KNOWLEDGE_BASE = [

    # ─────────────────────────────
    # COMSATS UNIVERSITY ISLAMABAD
    # ─────────────────────────────
    Document(page_content="""
University: COMSATS University Islamabad (CUI)
Type: Public Federal University
Ministry: Ministry of Science & Technology
Campuses: Islamabad, Lahore, Abbottabad, Wah, Attock, Vehari, Sahiwal
Website: www.comsats.edu.pk
Established: 2000
HEC Ranking: Ranked #1 in IT, #2 in Research in Pakistan
Total Students: 34,500+
PhD Faculty: 1,109+
""", metadata={"university": "COMSATS", "topic": "general"}),

    Document(page_content="""
University: COMSATS University Islamabad (CUI)
Topic: Undergraduate Admissions

Eligibility for BS Programs:
- Minimum 50% marks in FSc / A-Levels / ICS / equivalent qualification
- NTS or university entry test may be required for some programs
- Admissions twice a year: Fall (June) and Spring (October/November)

BS Programs offered:
- BS Computer Science
- BS Software Engineering
- BS Artificial Intelligence
- BS Data Science
- BS Cyber Security
- BS Electrical Engineering
- BS Computer Engineering
- BS Mathematics
- BS Physics
- BS Chemistry
- BS Biosciences / Bioinformatics
- BS Business Administration (BBA)
- BS Economics
- BS Finance & Accounting
- BS Psychology
- BS Media & Communication
- BS International Relations
- BS English
- BFA Fine Arts
- B.Arch Architecture
- BS Interior Design
- BS Remote Sensing & GIS

How to Apply:
- Apply online at: admissions.comsats.edu.pk
- Submit application form before deadline
- Upload transcripts, CNIC, passport photos
- Pay application fee online
""", metadata={"university": "COMSATS", "topic": "undergraduate admissions"}),

    Document(page_content="""
University: COMSATS University Islamabad (CUI)
Topic: Graduate (MS) Admissions

Eligibility for MS Programs:
- BS/BE degree (4 years) in relevant field
- Minimum 2.0 CGPA (on 4.0 scale) or 45% marks
- GAT General test with minimum 50% marks OR GRE equivalent
- NTS GAT is mandatory for most MS programs

MS Programs offered:
- MS Computer Science
- MS Software Engineering
- MS Artificial Intelligence
- MS Data Science
- MS Electrical Engineering
- MS Mathematics
- MS Physics
- MS Chemistry
- MS Environmental Sciences
- MS Biotechnology
- MS Business Administration (MBA)
- MS Management Sciences
- MS Economics
- MS Psychology
- MS Architecture
- MS Fine Arts

Admission Schedule:
- Fall: Applications open June, classes start September
- Spring: Applications open October/November, classes start February
- Spring 2026 deadline was January 3, 2026
""", metadata={"university": "COMSATS", "topic": "graduate ms admissions"}),

    Document(page_content="""
University: COMSATS University Islamabad (CUI)
Topic: PhD Admissions

Eligibility for PhD Programs:
- MS/MPhil degree in relevant field
- Minimum 3.0 CGPA (on 4.0 scale) or 60% marks
- GAT Subject test with minimum 60% marks OR GRE Subject test
- Research proposal may be required
- Interview with departmental committee

PhD Programs offered:
- PhD Computer Science
- PhD Electrical Engineering
- PhD Mathematics
- PhD Physics
- PhD Chemistry
- PhD Biosciences
- PhD Environmental Sciences
- PhD Management Sciences
- PhD Economics

PhD Duration: 3 to 5 years
Funding: HEC Indigenous Scholarships available for PhD students
""", metadata={"university": "COMSATS", "topic": "phd admissions"}),

    Document(page_content="""
University: COMSATS University Islamabad (CUI)
Topic: Fee Structure

BS Program Fees (per semester):
- Engineering programs: PKR 55,000 – PKR 75,000
- Computer Science / IT: PKR 55,000 – PKR 70,000
- Natural Sciences: PKR 45,000 – PKR 60,000
- Management Sciences: PKR 45,000 – PKR 60,000
- Social Sciences / Humanities: PKR 35,000 – PKR 50,000

MS Program Fees (per semester):
- Engineering / CS: PKR 60,000 – PKR 90,000
- Management / Sciences: PKR 55,000 – PKR 80,000

PhD Program Fees (per semester):
- All programs: PKR 65,000 – PKR 100,000

Note: Fees vary by campus. Islamabad campus may be slightly higher.
Scholarships: Need-based and merit-based scholarships available.
HEC need-based scholarships also accepted.
""", metadata={"university": "COMSATS", "topic": "fees"}),

    # ─────────────────────────────
    # NUST
    # ─────────────────────────────
    Document(page_content="""
University: National University of Sciences and Technology (NUST)
Type: Public Sector University (Federally Chartered)
Location: H-12, Islamabad, Pakistan
Website: www.nust.edu.pk
Established: 1991
HEC Ranking: Top 5 in Pakistan, ranked in QS World Rankings
Total Students: 15,000+
Constituent Colleges: SEECS, SCME, SNS, SMME, SADA, ASAB, S3H, NIPCONS, NBS, MCS, CAE, PNEC, CEME, NICE
""", metadata={"university": "NUST", "topic": "general"}),

    Document(page_content="""
University: NUST
Topic: Undergraduate Admissions

Eligibility for BE/BS Programs:
- Minimum 60% marks in FSc Pre-Engineering / Pre-Medical / A-Levels
- Mandatory entry test: NET (NUST Entry Test) — conducted by NUST itself
- NET score is the primary basis for merit
- No NTS required; NUST conducts its own test

Undergraduate Programs offered:
- BE Electrical Engineering
- BE Mechanical Engineering
- BE Civil Engineering
- BE Chemical Engineering
- BE Computer Engineering
- BE Software Engineering
- BE Avionics Engineering
- BE Aerospace Engineering
- BS Computer Science
- BS Artificial Intelligence
- BS Data Science
- BS Biosciences
- BS Environmental Sciences
- BS Mathematics
- BS Physics
- BS Chemistry
- BS Economics
- BBA (Business Administration)
- BS Accounting & Finance
- BS Media & Communication Studies
- BS Architecture
- BS Industrial Design

Admission Process:
- Register on nust.edu.pk for NET
- NET conducted multiple times per year (NET-1, NET-2, NET-3)
- Merit list based on NET score + FSc marks
- Provincial seats quota applies
- Admissions once a year (Fall semester only for most programs)
""", metadata={"university": "NUST", "topic": "undergraduate admissions"}),

    Document(page_content="""
University: NUST
Topic: Graduate (MS/MPhil) Admissions

Eligibility for MS Programs:
- 4-year BS/BE degree in relevant field
- Minimum 2.5 CGPA (on 4.0 scale)
- GAT General test with 50% marks OR GRE General (quantitative 145+)
- Some programs require GRE Subject test

MS Programs offered (selected):
- MS Electrical Engineering
- MS Mechanical Engineering
- MS Civil Engineering
- MS Computer Science
- MS Software Engineering
- MS Artificial Intelligence
- MS Data Science
- MS Biosciences
- MS Environmental Engineering
- MS Mathematics
- MS Physics
- MBA / MS Management Sciences
- MS Economics

Admissions: Spring (January) and Fall (August/September) semesters
Application Portal: nust.edu.pk/admissions
""", metadata={"university": "NUST", "topic": "graduate ms admissions"}),

    Document(page_content="""
University: NUST
Topic: PhD Admissions

Eligibility for PhD:
- MS/MPhil degree in relevant field
- Minimum 3.0 CGPA or 70% marks
- GAT Subject test 60% marks OR GRE Subject test
- Research proposal required
- Interview with supervisor/committee mandatory

PhD Programs: Available in all major engineering and science departments
PhD Duration: 3–5 years minimum
Funding: HEC Indigenous PhD Scholarships, NUST Research Assistantships available

Note: PhD students often work as Teaching/Research Assistants and receive stipends.
""", metadata={"university": "NUST", "topic": "phd admissions"}),

    Document(page_content="""
University: NUST
Topic: Fee Structure

BE/BS Program Fees (per semester):
- Engineering programs: PKR 145,000 – PKR 185,000
- Computer Science / IT: PKR 145,000 – PKR 175,000
- Natural / Social Sciences: PKR 120,000 – PKR 150,000
- Management / Business: PKR 130,000 – PKR 160,000

MS Program Fees (per semester):
- Engineering / CS: PKR 150,000 – PKR 190,000
- Management Sciences: PKR 140,000 – PKR 170,000

PhD Program Fees (per semester):
- All programs: PKR 90,000 – PKR 130,000 (lower due to research nature)

Hostel: Available on campus; separate hostel fees apply
Scholarships:
- NUST Merit Scholarship for top 10% students
- Need-based financial aid
- HEC scholarships accepted
- NUST offers fee waivers for high NET scorers
""", metadata={"university": "NUST", "topic": "fees"}),

    # ─────────────────────────────
    # UET LAHORE
    # ─────────────────────────────
    Document(page_content="""
University: University of Engineering and Technology (UET) Lahore
Type: Public Sector University (Provincial)
Location: Grand Trunk Road, Lahore, Punjab
Website: www.uet.edu.pk
Established: 1921 (oldest engineering university in Pakistan)
Campuses: Main campus Lahore, Narowal, Rachna, Kala Shah Kaku
HEC Ranking: Top engineering university in Punjab
""", metadata={"university": "UET Lahore", "topic": "general"}),

    Document(page_content="""
University: UET Lahore
Topic: Undergraduate Admissions

Eligibility for BE/BS Programs:
- Minimum 60% marks in FSc Pre-Engineering (Mathematics, Physics, Chemistry)
- Mandatory entry test: ECAT (Engineering College Admission Test) conducted by UET
- ECAT score + FSc marks = final merit
- Punjab domicile applicants get provincial seats

Undergraduate Programs offered:
- BE Civil Engineering
- BE Mechanical Engineering
- BE Electrical Engineering
- BE Chemical Engineering
- BE Computer Engineering
- BE Software Engineering
- BE Metallurgical & Materials Engineering
- BE Industrial Engineering
- BE Environmental Engineering
- BE Petroleum & Gas Engineering
- BS Architecture
- BS City & Regional Planning
- BS Computer Science
- BS Mathematics
- BS Physics
- BS Chemistry
- BS Food Engineering
- BE Agricultural Engineering
- BE Mechatronics Engineering

Admission Schedule:
- ECAT usually in August
- Merit list displayed September
- Classes start October/November
- Admissions once a year (Fall only)

Application: Apply online at www.uet.edu.pk
""", metadata={"university": "UET Lahore", "topic": "undergraduate admissions"}),

    Document(page_content="""
University: UET Lahore
Topic: Graduate (MS/MPhil) Admissions

Eligibility for MS Programs:
- 4-year BS/BE degree in relevant field
- Minimum 2.5 CGPA (on 4.0 scale) or 60% marks
- GAT General test with 50% marks OR GRE equivalent
- Written test / interview by department

MS Programs offered:
- MS Civil Engineering
- MS Structural Engineering
- MS Geotechnical Engineering
- MS Mechanical Engineering
- MS Electrical Engineering
- MS Power Engineering
- MS Computer Science
- MS Software Engineering
- MS Chemical Engineering
- MS Environmental Engineering
- MS Materials Engineering
- MS Industrial Engineering
- MS Mathematics
- MS Physics
- MS Chemistry

Admissions: Spring (February) and Fall (September) semesters
""", metadata={"university": "UET Lahore", "topic": "graduate ms admissions"}),

    Document(page_content="""
University: UET Lahore
Topic: PhD Admissions

Eligibility for PhD:
- MS/MPhil degree in relevant engineering or science field
- Minimum 3.0 CGPA or 60% marks in MS
- GAT Subject test with 60% marks OR GRE Subject test
- Research proposal submission required
- Departmental interview mandatory

PhD Programs: Available in all engineering and applied science departments
PhD Duration: 3–5 years
Funding: HEC Indigenous Scholarships, departmental research grants

Key strength: UET Lahore has strong industry connections in Lahore for research collaboration.
""", metadata={"university": "UET Lahore", "topic": "phd admissions"}),

    Document(page_content="""
University: UET Lahore
Topic: Fee Structure

BE/BS Program Fees (per semester):
- Engineering programs: PKR 45,000 – PKR 90,000
- Computer Science: PKR 45,000 – PKR 80,000
- Architecture / Planning: PKR 40,000 – PKR 70,000
- Natural Sciences: PKR 35,000 – PKR 60,000

Note: UET is a public university so fees are government-subsidized and lower than private universities.

MS Program Fees (per semester):
- Engineering / CS: PKR 55,000 – PKR 85,000

PhD Program Fees (per semester):
- All programs: PKR 50,000 – PKR 80,000

Hostel: On-campus hostels available for boys and girls. Separate hostel fee.
Scholarships:
- Punjab government merit scholarships
- HEC need-based scholarships
- University gold medals and awards for top students
""", metadata={"university": "UET Lahore", "topic": "fees"}),

    # ─────────────────────────────
    # QAU
    # ─────────────────────────────
    Document(page_content="""
University: Quaid-i-Azam University (QAU)
Type: Public Federal University
Location: Islamabad, Pakistan
Website: www.qau.edu.pk
Established: 1967
HEC Ranking: Ranked among Top 3 universities in Pakistan for research
Notable for: Strong research in Natural Sciences, Social Sciences, and Biosciences
""", metadata={"university": "QAU", "topic": "general"}),

    Document(page_content="""
University: QAU (Quaid-i-Azam University)
Topic: Undergraduate Admissions

Eligibility for BS Programs:
- Minimum 45%–50% marks in FSc / A-Levels / equivalent
- Entry test conducted by QAU or NTS
- Merit-based admissions

BS Programs offered:
- BS Mathematics
- BS Physics
- BS Chemistry
- BS Computer Science
- BS Biosciences
- BS Biochemistry
- BS Microbiology
- BS Biotechnology
- BS Environmental Sciences
- BS Statistics
- BS Economics
- BS Psychology
- BS Sociology
- BS Political Science
- BS International Relations
- BS History
- BS English Literature
- BS Gender Studies
- BS Anthropology
- BS Pakistan Studies
- BS Pharmacy (PharmD)
- BS Law (LLB)

Duration: 4 years (8 semesters)
Admissions: Once a year, Fall semester (August/September)
Application: Online via qau.edu.pk
""", metadata={"university": "QAU", "topic": "undergraduate admissions"}),

    Document(page_content="""
University: QAU (Quaid-i-Azam University)
Topic: Graduate (MS/MPhil) Admissions

Eligibility for MS/MPhil Programs:
- 4-year BS degree (or 2-year BSc + 2-year MSc) in relevant field
- Minimum 2.0 CGPA or 45% marks
- GAT General test with minimum 50% marks (mandatory — HEC requirement)
- Written departmental test and/or interview

MS/MPhil Programs offered:
- MS/MPhil Mathematics
- MS/MPhil Physics
- MS/MPhil Chemistry
- MS/MPhil Computer Science
- MS/MPhil Biosciences
- MS/MPhil Biochemistry
- MS/MPhil Microbiology
- MS/MPhil Biotechnology
- MS/MPhil Environmental Sciences
- MS/MPhil Statistics
- MS/MPhil Economics
- MS/MPhil Psychology
- MS/MPhil Sociology
- MS/MPhil Political Science
- MS/MPhil International Relations
- MS/MPhil History
- MS/MPhil Gender Studies

Admissions: Fall (August) and Spring (January/February) semesters
""", metadata={"university": "QAU", "topic": "graduate ms admissions"}),

    Document(page_content="""
University: QAU (Quaid-i-Azam University)
Topic: PhD Admissions

Eligibility for PhD:
- MPhil/MS degree in relevant field
- Minimum 3.0 CGPA or 60% marks
- GAT Subject test with minimum 60% marks
- Research proposal required
- Interview with departmental board

PhD Programs:
- Available in all departments (Sciences, Social Sciences, Biosciences, Pharmacy, Law)
- PhD Mathematics, Physics, Chemistry, Computer Science, Biosciences most popular

PhD Duration: 3–5 years
Funding:
- HEC Indigenous PhD Scholarships (fully funded)
- QAU Research Grants
- International research collaborations available

QAU is particularly strong in research — many faculty are HEC Distinguished Professors.
""", metadata={"university": "QAU", "topic": "phd admissions"}),

    Document(page_content="""
University: QAU (Quaid-i-Azam University)
Topic: Fee Structure

BS Program Fees (per semester):
- Natural Sciences (Physics, Chemistry, Math, Bio): PKR 15,000 – PKR 30,000
- Computer Science: PKR 20,000 – PKR 35,000
- Social Sciences / Humanities: PKR 12,000 – PKR 25,000
- Pharmacy (PharmD): PKR 35,000 – PKR 55,000
- Law (LLB): PKR 20,000 – PKR 35,000

Note: QAU is a federal public university with heavily subsidized fees — among the lowest in Pakistan.

MS/MPhil Fees (per semester):
- All programs: PKR 20,000 – PKR 45,000

PhD Fees (per semester):
- All programs: PKR 25,000 – PKR 50,000

Hostel: On-campus hostels available for boys and girls at low cost
Scholarships:
- HEC need-based scholarships
- Federal government merit scholarships
- Departmental assistantships for MS/PhD students
""", metadata={"university": "QAU", "topic": "fees"}),

    # ─────────────────────────────
    # COMPARISON DOCS
    # ─────────────────────────────
    Document(page_content="""
Topic: Comparison of 4 Pakistani Universities

Fee Comparison (BS per semester, approximate):
- QAU: PKR 12,000 – PKR 55,000 (cheapest, federal public university)
- UET Lahore: PKR 35,000 – PKR 90,000 (affordable public university)
- COMSATS: PKR 35,000 – PKR 75,000 (public but self-financed)
- NUST: PKR 120,000 – PKR 185,000 (most expensive among the four)

Entry Test Comparison:
- COMSATS: NTS GAT or university test
- NUST: NET (NUST's own test — most competitive)
- UET Lahore: ECAT (UET's own test)
- QAU: NTS or departmental test

Minimum FSc Marks:
- QAU: 45%–50%
- COMSATS: 50%
- UET Lahore: 60%
- NUST: 60% (and high NET score required)

Scholarship Availability: All four offer HEC need-based scholarships.

Best for Engineering: NUST, UET Lahore
Best for Sciences: QAU, COMSATS
Best for IT/CS: NUST, COMSATS
Best Affordable Option: QAU (lowest fees), UET Lahore (public engineering)
""", metadata={"university": "All", "topic": "comparison"}),

    Document(page_content="""
Topic: PhD Eligibility Summary for All 4 Universities

COMSATS PhD:
- MS degree, 3.0 CGPA minimum
- GAT Subject 60% or GRE Subject
- Research proposal + interview

NUST PhD:
- MS degree, 3.0 CGPA minimum, 70% marks preferred
- GAT Subject 60% or GRE Subject
- Research proposal + supervisor interview

UET Lahore PhD:
- MS/MPhil in engineering or science
- 3.0 CGPA or 60% marks
- GAT Subject 60%
- Research proposal + departmental interview

QAU PhD:
- MPhil/MS degree, 3.0 CGPA or 60% marks
- GAT Subject 60%
- Research proposal + board interview

Common requirement for all: HEC GAT Subject test is mandatory for PhD admissions in Pakistan.
HEC Indigenous PhD Scholarship is available at all four universities for deserving candidates.
""", metadata={"university": "All", "topic": "phd comparison"}),

    Document(page_content="""
Topic: Scholarships at Pakistani Universities

HEC Need-Based Scholarship:
- Available at all public universities including COMSATS, NUST, UET, QAU
- Covers tuition fee partially or fully
- Based on family income (below PKR 45,000/month household income)
- Apply via HEC portal: hec.gov.pk

HEC Indigenous PhD Scholarship:
- Fully funded for Pakistani PhD students
- Covers tuition + monthly stipend
- Available at all 4 universities

NUST Merit Scholarship:
- For top 10% students based on CGPA
- Partial to full tuition waiver

UET Punjab Government Scholarship:
- For Punjab domicile students with financial need

COMSATS Need-Based Aid:
- Internal university scholarship
- Apply at time of admission

QAU Federal Scholarship:
- For students from underprivileged areas
- Hostel accommodation included
""", metadata={"university": "All", "topic": "scholarships"}),
]


# ==============================
# BUILD VECTORSTORE
# ==============================
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    split_docs = splitter.split_documents(KNOWLEDGE_BASE)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


print("Loading knowledge base...")
vectorstore = build_vectorstore()
print("Knowledge base ready.")


# ==============================
# CHAT FUNCTION
# gr.ChatInterface passes history as a list of dicts automatically
# ==============================
def chat(user_message, history):
    try:
        docs = vectorstore.similarity_search(user_message, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a helpful university admissions assistant for Pakistani students.
You have detailed knowledge about these 4 universities:
1. COMSATS University Islamabad (CUI)
2. NUST - National University of Sciences and Technology
3. UET Lahore - University of Engineering and Technology Lahore
4. QAU - Quaid-i-Azam University Islamabad

Use the context below to answer the student's question clearly and accurately.
Use bullet points for lists. Be specific with numbers, percentages, and requirements.
If the question is about a university not in your knowledge base, politely say you only cover these 4 universities.
If specific information is not in the context, say so honestly and suggest checking the official website.

Context:
{context}

Student Question: {user_message}

Answer:"""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"


# ==============================
# GRADIO UI — using ChatInterface
# Works on ALL Gradio versions (no type= argument needed)
# ==============================
demo = gr.ChatInterface(
    fn=chat,
    title="🎓 Pakistan University Assistant",
    description=(
        "### Your guide to admissions, fees, programs & scholarships\n"
        "**Covered Universities:** COMSATS · NUST · UET Lahore · QAU\n\n"
        "*Always verify details on official university websites before applying.*"
    ),
    examples=[
        "What is the eligibility criteria for PhD Mathematics at QAU?",
        "Compare fees of NUST and UET Lahore for BS Computer Science",
        "What entry test is required for COMSATS undergraduate admissions?",
        "What scholarships are available for MS students in Pakistan?",
        "What programs does QAU offer in Social Sciences?",
        "What is the fee structure for BS Electrical Engineering at UET Lahore?",
        "When does NUST take admissions for BS programs?",
        "What is the minimum CGPA required for PhD at COMSATS?",
    ],
)

demo.launch()