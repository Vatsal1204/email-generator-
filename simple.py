"""
email_simple.py - Stunning Modern UI Version (Same Logic)
Run: streamlit run email_simple.py
"""

import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import base64

# ============================================
# PAGE CONFIG - MUST BE FIRST
# ============================================
st.set_page_config(
    page_title="Email AI Studio", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ADVANCED UI STYLING
# ============================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* User avatar */
    .user-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        border: 3px solid white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .user-avatar span {
        color: white;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .step-container::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: rgba(255,255,255,0.1);
        transform: translateY(-50%);
        z-index: 1;
    }
    
    .step-item {
        background: rgba(255,255,255,0.95);
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        position: relative;
        z-index: 2;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .step-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: white;
        transform: scale(1.05);
    }
    
    .step-item.completed {
        background: #10b981;
        color: white;
    }
    
    /* Intent cards */
    .intent-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: all 0.3s ease;
    }
    
    .intent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
    }
    
    /* Progress bars */
    .prob-bar-container {
        background: #f0f0f0;
        border-radius: 10px;
        height: 30px;
        margin: 10px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .prob-bar {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        transition: width 1s ease;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Email preview */
    .email-preview {
        background: #1e1e2e;
        color: #fff;
        padding: 2rem;
        border-radius: 15px;
        font-family: 'Courier New', monospace;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        white-space: pre-wrap;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text area */
    .stTextArea > div > div > textarea {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE - SAME LOGIC
# ============================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'intent' not in st.session_state:
    st.session_state.intent = None
if 'message' not in st.session_state:
    st.session_state.message = ""
if 'company' not in st.session_state:
    st.session_state.company = ""
if 'rec_name' not in st.session_state:
    st.session_state.rec_name = ""
if 'rec_email' not in st.session_state:
    st.session_state.rec_email = ""
if 'rec_company' not in st.session_state:
    st.session_state.rec_company = ""
if 'rec_subject' not in st.session_state:
    st.session_state.rec_subject = ""
if 'email_body' not in st.session_state:
    st.session_state.email_body = ""

# ============================================
# LOAD MODEL - SAME LOGIC
# ============================================
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load("ultra_fast_model.pt", weights_only=False, map_location='cpu')
        encoder = checkpoint['label_encoder']
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(encoder.classes_)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        return model, tokenizer, encoder
    except Exception as e:
        st.error(f"⚠️ Model not found: {e}")
        return None, None, None

# ============================================
# LOGIN PAGE - STUNNING UI
# ============================================
def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">✨</h1>
            <h1 class="gradient-text" style="font-size: 3rem; margin-bottom: 0;">Email AI Studio</h1>
            <p style="color: #666; font-size: 1.2rem;">Intelligent Email Automation</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            with st.form("login_form"):
                st.markdown("### 🔐 Welcome Back")
                email = st.text_input("Email Address", placeholder="your@email.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_btn = st.form_submit_button("Login", type="primary", use_container_width=True)
                with col_b:
                    demo_btn = st.form_submit_button("Demo", use_container_width=True)
            
            if login_btn and email:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.rerun()
            
            if demo_btn:
                st.session_state.logged_in = True
                st.session_state.user_email = "demo@email.ai"
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - STUNNING UI
# ============================================
def sidebar():
    with st.sidebar:
        # User avatar
        initial = st.session_state.user_email[0].upper() if st.session_state.user_email else "U"
        st.markdown(f"""
        <div class="user-avatar">
            <span>{initial}</span>
        </div>
        <h3 style="text-align: center; color: white; margin-bottom: 0;">{st.session_state.user_email.split('@')[0]}</h3>
        <p style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.9rem;">{st.session_state.user_email}</p>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # Navigation
        if st.button("✨ Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
        if st.button("📝 Compose", use_container_width=True):
            st.session_state.step = 1
            st.session_state.page = "compose"
            st.rerun()
        if st.button("📊 History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

# ============================================
# STEP 1: WRITE - STUNNING UI
# ============================================
def step_write():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 2.5rem;">✍️ Compose Your Message</h1>
        <p style="color: #666;">Write naturally - our AI will understand your intent</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        with st.form("write_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Your Name", value=st.session_state.get('user_name', ''), 
                                    placeholder="John Doe")
            with col2:
                company = st.text_input("Your Company", placeholder="ABC Corp",
                                       value=st.session_state.get('company', ''))
            
            message = st.text_area(
                "Your Message",
                height=150,
                placeholder="e.g., We see strong potential for collaboration between our companies. Let's explore partnership opportunities in the Mumbai region.",
                value=st.session_state.get('message', '')
            )
            
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            if st.form_submit_button("🔍 Analyze Intent", type="primary", use_container_width=True):
                if message:
                    st.session_state.user_name = name
                    st.session_state.company = company
                    st.session_state.message = message
                    st.session_state.step = 2
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# STEP 2: ANALYZE - STUNNING UI
# ============================================
def step_analyze():
    model, tokenizer, encoder = load_model()
    
    if model and st.session_state.message:
        inputs = tokenizer(st.session_state.message, truncation=True, padding=True, 
                          max_length=64, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        intent = encoder.classes_[pred]
        confidence = probs[pred].item()
        st.session_state.intent = intent
        
        # Color mapping
        colors = {
            'inquiry': '#2196F3',
            'complaint': '#F44336', 
            'sales': '#4CAF50',
            'negotiation': '#FF9800',
            'partnership': '#9C27B0'
        }
        color = colors.get(intent, '#667eea')
        
        # Show intent card
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text" style="font-size: 2rem;">🎯 Intent Analysis</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="intent-card" style="border-left-color: {color};">
                <h2 style="color: {color}; font-size: 2rem; margin-bottom: 0.5rem;">{intent.upper()}</h2>
                <div style="font-size: 1.5rem; font-weight: 600; margin: 1rem 0;">
                    <span style="color: {color};">{confidence:.1%}</span>
                </div>
                <div style="background: {color}20; padding: 1rem; border-radius: 10px;">
                    <p style="color: #666; font-style: italic;">"{st.session_state.message[:100]}..."</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Probability distribution
        st.markdown("### 📊 Probability Distribution")
        probs_dict = {encoder.classes_[i]: probs[i].item() for i in range(len(encoder.classes_))}
        
        for label, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{label.title()}**")
            with col2:
                bar_color = colors.get(label, '#667eea')
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: {prob*100}%; background: {bar_color};">
                        {prob:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Correct", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col3:
            if st.button("✏️ Edit", use_container_width=True):
                st.session_state.step = 1
                st.rerun()

# ============================================
# STEP 3: RECIPIENT - STUNNING UI
# ============================================
def step_recipient():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 2rem;">👤 Recipient Details</h1>
        <p style="color: #666;">Who should receive this email?</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        with st.form("recipient_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Recipient Name", placeholder="John Smith")
            with col2:
                company = st.text_input("Company Name", placeholder="ABC Corp")
            
            email = st.text_input("Email Address", placeholder="john@abccorp.com")
            subject = st.text_input(
                "Subject",
                value=f"Regarding: {st.session_state.intent.title() if st.session_state.intent else 'Inquiry'}"
            )
            
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            if st.form_submit_button("✍️ Generate Email", type="primary", use_container_width=True):
                if name and email:
                    st.session_state.rec_name = name
                    st.session_state.rec_company = company
                    st.session_state.rec_email = email
                    st.session_state.rec_subject = subject
                    st.session_state.step = 4
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# STEP 4: REVIEW & SEND - STUNNING UI
# ============================================
def step_review():
    # Email templates (SAME LOGIC)
    templates = {
        "inquiry": f"""Dear {st.session_state.rec_name},

I hope this email finds you well.

{st.session_state.message}

Could you please share your catalog and pricing information at your earliest convenience?

Thank you for your time and consideration.

Best regards,
{st.session_state.user_name}
{st.session_state.company}
{st.session_state.user_email}""",
        
        "complaint": f"""Dear {st.session_state.rec_name},

I am writing to bring an important matter to your attention.

{st.session_state.message}

Please look into this matter urgently and let me know how you plan to resolve this issue.

I look forward to your prompt response.

Regards,
{st.session_state.user_name}
{st.session_state.company}
{st.session_state.user_email}""",
        
        "sales": f"""Dear {st.session_state.rec_name},

I hope you're doing well.

{st.session_state.message}

Would you be available for a quick 15-minute call next week to discuss how we might work together? Please let me know what time works best for you.

Looking forward to connecting!

Best regards,
{st.session_state.user_name}
{st.session_state.company}
{st.session_state.user_email}""",
        
        "negotiation": f"""Dear {st.session_state.rec_name},

Thank you for your proposal.

{st.session_state.message}

We're very interested in moving forward, but would appreciate if you could reconsider the pricing. Could we schedule a brief call to discuss this further?

Thank you for your understanding.

Regards,
{st.session_state.user_name}
{st.session_state.company}
{st.session_state.user_email}""",
        
        "partnership": f"""Dear {st.session_state.rec_name},

I've been following {st.session_state.rec_company}'s impressive work in the industry.

{st.session_state.message}

I see great potential for collaboration between our companies. Would you be open to an exploratory conversation to discuss possible synergies?

I look forward to hearing from you.

Best regards,
{st.session_state.user_name}
{st.session_state.company}
{st.session_state.user_email}"""
    }
    
    email_body = templates.get(st.session_state.intent, templates['inquiry'])
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 2rem;">📨 Review & Send</h1>
        <p style="color: #666;">Your AI-generated email is ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Email preview
    st.markdown("### 📧 Generated Email")
    st.markdown(f'<div class="email-preview">{email_body.replace(chr(10), "<br>")}</div>', 
                unsafe_allow_html=True)
    
    # Edit option
    edited_email = st.text_area("✏️ Edit if needed:", value=email_body, height=200)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Send section
    st.markdown("### 📤 Send Email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        password = st.text_input(
            "🔑 App Password",
            type="password",
            help="Enter your 16-character Gmail App Password",
            placeholder="••••••••••••••••"
        )
        
        if st.button("📤 Send Now", type="primary", use_container_width=True):
            if password:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = st.session_state.user_email
                    msg['To'] = st.session_state.rec_email
                    msg['Subject'] = st.session_state.rec_subject
                    msg.attach(MIMEText(edited_email, 'plain'))
                    
                    server = smtplib.SMTP("smtp.gmail.com", 587)
                    server.starttls()
                    server.login(st.session_state.user_email, password)
                    server.send_message(msg)
                    server.quit()
                    
                    st.success("✅ Email sent successfully! 🎉")
                    st.balloons()
                    
                    if st.button("📝 Compose New", key="new_after_send"):
                        st.session_state.step = 1
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {str(e)}")
            else:
                st.error("❌ Please enter your app password")
    
    with col2:
        if st.button("💾 Save Draft", use_container_width=True):
            st.success("✅ Draft saved!")
        if st.button("🔄 Edit Recipient", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    if st.button("📝 Start Fresh", use_container_width=True):
        st.session_state.step = 1
        st.rerun()

# ============================================
# DASHBOARD - STUNNING UI
# ============================================
def dashboard():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="gradient-text" style="font-size: 3rem;">✨ Dashboard</h1>
        <p style="color: #666; font-size: 1.2rem;">Your email activity at a glance</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">12</div>
            <p style="color: #666;">Emails Sent</p>
            <p style="color: #10b981; font-size: 0.9rem;">↑ +3 this week</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">8</div>
            <p style="color: #666;">Responses</p>
            <p style="color: #10b981; font-size: 0.9rem;">67% response rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">5</div>
            <p style="color: #666;">Meetings</p>
            <p style="color: #f59e0b; font-size: 0.9rem;">2 scheduled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">3</div>
            <p style="color: #666;">Partnerships</p>
            <p style="color: #10b981; font-size: 0.9rem;">+1 this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    
    activities = [
        {"time": "2 hours ago", "action": "Partnership proposal to TechCorp", "status": "✅ Sent"},
        {"time": "Yesterday", "action": "Sales demo request to Innovate Inc", "status": "✅ Sent"},
        {"time": "2 days ago", "action": "Inquiry about products", "status": "✅ Replied"},
    ]
    
    for act in activities:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div>
                <strong>{act['action']}</strong><br>
                <small style="color: #999;">{act['time']}</small>
            </div>
            <span>{act['status']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    if st.button("📝 Compose New Email", type="primary"):
        st.session_state.step = 1
        st.session_state.page = "compose"
        st.rerun()

# ============================================
# HISTORY PAGE
# ============================================
def history():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="gradient-text" style="font-size: 2.5rem;">📊 Email History</h1>
        <p style="color: #666;">Track all your sent emails</p>
    </div>
    """, unsafe_allow_html=True)
    
    import pandas as pd
    data = {
        'Date': ['2024-02-22', '2024-02-21', '2024-02-20', '2024-02-19'],
        'To': ['john@techcorp.com', 'sarah@innovate.com', 'support@supplyco.com', 'info@global.com'],
        'Intent': ['Sales', 'Partnership', 'Complaint', 'Inquiry'],
        'Status': ['✅ Sent', '✅ Sent', '✅ Sent', '✅ Sent']
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# ============================================
# MAIN APP - SAME LOGIC
# ============================================
def main():
    if not st.session_state.logged_in:
        login()
        return
    
    sidebar()
    
    # Progress steps (only for compose page)
    if st.session_state.get('page') == 'compose':
        st.markdown("""
        <div class="step-container">
            <div class="step-item {}">✍️ Write</div>
            <div class="step-item {}">🔍 Analyze</div>
            <div class="step-item {}">👤 Recipient</div>
            <div class="step-item {}">📨 Review</div>
        </div>
        """.format(
            'active' if st.session_state.step == 1 else 'completed' if st.session_state.step > 1 else '',
            'active' if st.session_state.step == 2 else 'completed' if st.session_state.step > 2 else '',
            'active' if st.session_state.step == 3 else 'completed' if st.session_state.step > 3 else '',
            'active' if st.session_state.step == 4 else 'completed' if st.session_state.step > 4 else ''
        ), unsafe_allow_html=True)
        
        # Show current step
        if st.session_state.step == 1:
            step_write()
        elif st.session_state.step == 2:
            step_analyze()
        elif st.session_state.step == 3:
            step_recipient()
        elif st.session_state.step == 4:
            step_review()
    else:
        if st.session_state.get('page') == 'dashboard':
            dashboard()
        elif st.session_state.get('page') == 'history':
            history()
        else:
            dashboard()  # default

if __name__ == "__main__":
    main()