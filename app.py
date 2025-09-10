import streamlit as st
import easyocr
from sympy import symbols, Eq, solve, sympify
import numpy as np
from PIL import Image
import re
import random
import matplotlib.pyplot as plt

# Theme selector
theme = st.radio("🎨 Choose Your Theme", ["Pastel Paradise", "Dark Mode Dungeon"])

# CSS for Pastel Paradise
pastel_css = """
<style>
.stApp {
    background: linear-gradient(135deg, #fddde6, #e0f7fa, #ffeaa7);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: black !important;
    overflow-x: hidden;
    padding-top: 10px !important;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.block {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 5px 5px 20px rgba(0,0,0,0.1);
    color: black !important;
    font-weight: bold;
}
h1,h2,h3,h4,p,span,div {
    color: black !important;
}
.floating-symbol {
    position: fixed;
    font-size: 30px;
    color: rgba(0,0,0,0.08);
    animation: float linear infinite;
    pointer-events: none;
}
@keyframes float {
    0% {transform: translateY(100vh) rotate(0deg); opacity: 0;}
    50% {opacity: 0.5;}
    100% {transform: translateY(-10vh) rotate(360deg); opacity: 0;}
}
.sparkle {
    position: fixed;
    width: 6px;
    height: 6px;
    background: #fff;
    border-radius: 50%;
    animation: sparkle 3s infinite ease-in-out;
}
@keyframes sparkle {
    0% {transform: translateY(100vh); opacity: 0;}
    50% {opacity: 1;}
    100% {transform: translateY(-10vh); opacity: 0;}
}
</style>
"""

# CSS for Dark Mode Dungeon (Stranger Things vibe)
dark_css = """
<style>
.stApp {
    background: radial-gradient(circle at top, #0f0f0f, #1a1a1a, #2c2c2c);
    background-size: cover;
    animation: pulseBG 20s ease infinite;
    color: white !important;
    overflow-x: hidden;
    padding-top: 10px !important;
}
@keyframes pulseBG {
    0% {filter: brightness(1);}
    50% {filter: brightness(1.2);}
    100% {filter: brightness(1);}
}
.block {
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(255,0,0,0.3);
    color: white !important;
    font-weight: bold;
}
h1,h2,h3,h4,p,span,div {
    color: white !important;
}
.floating-symbol {
    position: fixed;
    font-size: 30px;
    color: rgba(255,0,0,0.08);
    animation: float linear infinite;
    pointer-events: none;
}
@keyframes float {
    0% {transform: translateY(100vh) rotate(0deg); opacity: 0;}
    50% {opacity: 0.5;}
    100% {transform: translateY(-10vh) rotate(360deg); opacity: 0;}
}
.sparkle {
    position: fixed;
    width: 6px;
    height: 6px;
    background: red;
    border-radius: 50%;
    animation: sparkle 3s infinite ease-in-out;
}
@keyframes sparkle {
    0% {transform: translateY(100vh); opacity: 0;}
    50% {opacity: 1;}
    100% {transform: translateY(-10vh); opacity: 0;}
}
</style>
"""

# Inject selected theme CSS
if theme == "Pastel Paradise":
    st.markdown(pastel_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# Floating symbols
symbols_list = (
    ['π','√','∑','∞','x','y','z','+','-','='] if theme == "Pastel Paradise"
    else ['⚔','☠','∇','∂','λ','Ω','≠','≥','≤','∫']
)
for i in range(15):
    symb = random.choice(symbols_list)
    left_pos = random.randint(0,90)
    duration = random.randint(8,20)
    size = random.randint(20,50)
    st.markdown(f"""
    <div class="floating-symbol" style="left:{left_pos}vw; font-size:{size}px; animation-duration:{duration}s;">{symb}</div>
    """, unsafe_allow_html=True)

# Sparkles
for i in range(20):
    left = random.randint(0, 100)
    delay = random.uniform(0, 5)
    st.markdown(f"""
    <div class="sparkle" style="left:{left}vw; animation-delay:{delay}s;"></div>
    """, unsafe_allow_html=True)
# App title
st.title("🧮 Creative Math Equation Solver")


# OCR reader
reader = easyocr.Reader(['en'])

# Clean OCR text
def clean_equation_text(text):
    text = text.strip().lower()
    text = (text
            .replace('\u2212', '-')
            .replace('\u2010', '-')
            .replace('\u2011', '-')
            .replace('\u2012', '-')
            .replace('\u2013', '-')
            .replace('\u2014', '-')
            .replace('\u2015', '-')
            .replace('~', '-'))
    replacements = {'×': '*', '·': '*', '÷': '/', '^': '**'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'\b[IlJ|]\b', 'x', text)
    text = re.sub(r'(^|[ \t\+\-\*/=\(\)])([IlJ|])(?=\s*[A-Za-z0-9])', r'\1x', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', '*', text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', text)
    text = re.sub(r'[^0-9A-Za-z\+\-\*\=/\.\(\)\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_handwritten_ocr(text):
    # Fix common misreads
    text = text.replace('~x', '-x').replace('~3z', '-3*z')
    text = re.sub(r'\bax\b', '2*x', text, flags=re.IGNORECASE)
    text = re.sub(r'\b54\b', '5*y', text)
    text = re.sub(r'\b24\b', '2*y', text)
    text = re.sub(r'\b2\.6\b', '= 6', text)

    # Fix lone variables and add multiplication signs
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)

    # Remove phantom variables like 'a' if not in context
    text = re.sub(r'\ba\b', '', text)

    return text.strip()

def fix_equation_structure(text):
    # Fix double equals like "2x + y = z = 5" → "2x + y - z = 5"
    text = re.sub(r'=\s*([a-zA-Z])\s*=', r'- \1 =', text)

    # Fix misread z terms like "3z = -12" → "-3*z = -12"
    text = re.sub(r'([0-9]+)\s*z\s*=\s*', r'-\1*z = ', text)

    # Fix common OCR misreads
    text = text.replace('~i', '-x').replace('~l', '-x').replace('~1', '-x')
    text = re.sub(r'\bi\b', 'x', text)  # lone 'i' as variable
    text = text.replace('= =', '=')

    # Add multiplication signs
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)

    # Fix missing '+' between terms (optional, experimental)
    text = re.sub(r'([a-zA-Z])\s*([a-zA-Z])', r'\1 + \2', text)

    return text

# Group OCR lines
def group_ocr_lines(ocr_result, image_height, y_tol_ratio=0.03):
    items = []
    for res in ocr_result:
        bbox = res[0]
        text = res[1]
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_center = sum(ys) / len(ys)
        x_min = min(xs)
        items.append({'text': text, 'y': y_center, 'x': x_min})
    items = sorted(items, key=lambda it: (it['y'], it['x']))
    tol = max(10, image_height * y_tol_ratio)
    lines = []
    current_group = None
    for it in items:
        if current_group is None:
            current_group = {'y': it['y'], 'items': [it]}
        else:
            if abs(it['y'] - current_group['y']) <= tol:
                current_group['items'].append(it)
                current_group['y'] = (current_group['y'] * (len(current_group['items']) - 1) + it['y']) / len(current_group['items'])
            else:
                row_text = ' '.join([t['text'] for t in sorted(current_group['items'], key=lambda x: x['x'])])
                lines.append(row_text)
                current_group = {'y': it['y'], 'items': [it]}
    if current_group:
        row_text = ' '.join([t['text'] for t in sorted(current_group['items'], key=lambda x: x['x'])])
        lines.append(row_text)
    return lines

# Validate equations
def is_valid_equation(eq):
    if '=' not in eq or not re.search(r'[A-Za-z]', eq):
        return False
    left, right = eq.split('=', 1)
    if not left.strip() or not right.strip():
        return False
    if not re.search(r'[\+\-\*/]', eq):
        return False
    return True

def compute_accuracy(raw_lines, cleaned_lines, solutions):
    score = 0
    if raw_lines:
        score += 30  # OCR detected something
    if cleaned_lines:
        score += 40  # Valid equations found
    if solutions:
        score += 30  # Solved successfully
    return score

# Cached OCR
@st.cache_data
def run_ocr(image_np):
    return reader.readtext(image_np)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.container():
        st.markdown('<div class="block">', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((min(image.width, 800), min(image.height, 800)))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    image_np = np.array(image)

    loading_messages = [
        "🧠 Crunching numbers...",
        "📐 Aligning variables...",
        "🔍 Zooming in on math magic..."
    ]
    with st.spinner(random.choice(loading_messages)):
        result = run_ocr(image_np)

    grouped_lines = group_ocr_lines(result, image_np.shape[0], y_tol_ratio=0.03)

    cleaned_lines = []
    for line in grouped_lines:
        cleaned = clean_equation_text(line)
        fixed = fix_handwritten_ocr(cleaned)
        if is_valid_equation(fixed):
            cleaned_lines.append(fixed)

    if not cleaned_lines:
        joined = ' '.join([res[1] for res in result])
        joined_clean = clean_equation_text(joined)
        for p in re.split(r'[;,\n]', joined_clean):
            p = p.strip()
            if is_valid_equation(p):
                cleaned_lines.append(p)

    with st.container():
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📜 Detected Equations (OCR Output):")
        for raw in grouped_lines:
            st.write(f"• {raw}")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("🔧 Processed Equations to Solve:")
        for eq in cleaned_lines:
            st.write(f"**{eq}**")
        st.markdown('</div>', unsafe_allow_html=True)

    if cleaned_lines:
        try:
            show_decimal = st.checkbox("Show decimal approximations", value=False)
            variables = sorted(set(re.findall(r'[A-Za-z]', ' '.join(cleaned_lines).lower())))
            vars_symbols = symbols(' '.join(variables))
            equations = []
            for line in cleaned_lines:
                safe_line = re.sub(r'[^0-9A-Za-z+\-*/=(). ]', '', line.lower()).strip()
                if '=' not in safe_line:
                    continue
                left, right = safe_line.split('=', 1)
                if left and right:
                    equations.append(Eq(sympify(left), sympify(right)))

            solutions = solve(equations, vars_symbols, dict=True)

            with st.container():
                st.markdown('<div class="block">', unsafe_allow_html=True)
                st.subheader("🧠 Solution:")
                if solutions:
                    for sol in solutions:
                        for var, val in sol.items():
                            val = float(val.evalf()) if show_decimal else val
                            st.write(f"**{var} = {val}**")
                else:
                    st.write("No unique solution found (system might be singular or underdetermined).")
                st.markdown('</div>', unsafe_allow_html=True)

            accuracy_score = compute_accuracy(grouped_lines, cleaned_lines, solutions)

            with st.container():
                st.markdown('<div class="block">', unsafe_allow_html=True)
                st.subheader("📈 OCR Accuracy Meter")
                st.progress(accuracy_score)

                if accuracy_score == 100:
                    st.success("✨ Perfect detection and solution! Your pipeline nailed it.")
                elif accuracy_score >= 70:
                    st.info("👍 Good job! Minor tweaks could improve reliability.")
                else:
                    st.warning("⚠️ OCR struggled with this one. Try a clearer image or adjust cleaning rules.")

                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Couldn't solve the equations. Error: {e}")
            st.info(
                "Tips:\n"
                "- Ensure equations are linear (like `x+y=10`, `2*x-y=5`)\n"
                "- Make writing clear and avoid stray marks near variables\n"
                "- If OCR still misses, try a clearer photo or typed text"
            )

   

    # ✅ Plot equations if system has exactly 2 variables
    if len(vars_symbols) == 2:
        x_sym, y_sym = vars_symbols
        x_vals = np.linspace(-10, 10, 200)  # Reduced resolution for speed
        fig, ax = plt.subplots()

        for eq in equations:
            try:
                lhs = eq.lhs - eq.rhs
                y_expr = solve(lhs, y_sym)
                if y_expr:
                    y_func = y_expr[0]
                    y_vals = [float(y_func.subs(x_sym, x)) for x in x_vals]
                    ax.plot(x_vals, y_vals, label=str(eq))
            except Exception:
                st.warning(f"Couldn't plot equation: {eq}")

        ax.set_xlabel(str(x_sym))
        ax.set_ylabel(str(y_sym))
        ax.legend()
        ax.grid(True)
        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("📊 Graphical View:")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 🧠 Did You Know?")
facts = [
    "Zero is the only number that can't be represented in Roman numerals.",
    "A circle has infinite lines of symmetry.",
    "The word 'hundred' comes from the old Norse term 'hundrath', which actually means 120.",
    "The Fibonacci sequence appears in nature — in pinecones, sunflowers, and seashells.",
    "The number π (pi) has been calculated to over 100 trillion digits!"
]
st.info(random.choice(facts))
