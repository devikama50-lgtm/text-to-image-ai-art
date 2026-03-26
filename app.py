import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Art Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================
os.makedirs("outputs", exist_ok=True)

# ============================================
# SESSION STATE
# ============================================
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "prompts_history" not in st.session_state:
    st.session_state.prompts_history = []

# ============================================
# LOAD MODEL WITH CACHING
# ============================================
@st.cache_resource
def load_pipeline():
    """Load Stable Diffusion pipeline with caching"""
    
    with st.spinner("📦 Loading AI model... This may take 1-2 minutes on first run"):
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # Use faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        
        # Memory optimization
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    return pipe

# ============================================
# GENERATE IMAGE FUNCTION
# ============================================
def generate_image(prompt, negative_prompt, steps, guidance_scale, height, width, seed):
    """Generate image from prompt"""
    
    if not prompt.strip():
        st.error("❌ Please enter a prompt!")
        return None
    
    try:
        # Set seed for reproducibility
        if seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        with torch.autocast("cuda"):
            result = st.session_state.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            )
        
        return result.images[0]
    
    except Exception as e:
        st.error(f"❌ Error generating image: {str(e)}")
        return None

# ============================================
# SAVE IMAGE FUNCTION
# ============================================
def save_image(image, prompt):
    """Save generated image to outputs folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/art_{timestamp}.png"
    image.save(filename)
    return filename

# ============================================
# HEADER & TITLE
# ============================================
st.title("🎨 AI Art Generator")
st.subheader("Create stunning images from text descriptions using Stable Diffusion")
st.divider()

# ============================================
# LOAD MODEL
# ============================================
st.session_state.pipeline = load_pipeline()

# ============================================
# MAIN LAYOUT - TWO COLUMNS
# ============================================
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    st.subheader("✨ Create Your Artwork")
    
    # ====== MAIN PROMPT INPUT ======
    prompt = st.text_area(
        "📝 Describe your image:",
        value="A serene fantasy landscape with floating islands, magical aurora lights, cinematic lighting",
        height=120,
        help="Be specific! Include subject, style, lighting, and quality details"
    )
    
    # ====== ADVANCED OPTIONS ======
    with st.expander("⚙️ Advanced Settings", expanded=False):
        
        # Two columns for settings
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("**Quality Settings**")
            steps = st.slider(
                "Inference Steps:",
                min_value=10,
                max_value=80,
                value=30,
                step=5,
                help="Higher = better quality but slower (20-50 recommended)"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale:",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="How strictly to follow the prompt (7-12 recommended)"
            )
        
        with adv_col2:
            st.markdown("**Image Size**")
            height = st.selectbox(
                "Height:",
                options=[256, 384, 512, 640, 768],
                index=2,
                help="512x512 is default"
            )
            
            width = st.selectbox(
                "Width:",
                options=[256, 384, 512, 640, 768],
                index=2
            )
        
        # Negative prompt
        st.markdown("**What to Avoid**")
        negative_prompt = st.text_area(
            "Negative Prompt:",
            value="blurry, low quality, distorted, ugly, deformed",
            height=60,
            help="Describe what you DON'T want in the image"
        )
        
        # Seed
        seed = st.number_input(
            "Seed (for reproducibility):",
            value=-1,
            help="-1 = random, use same number to recreate similar images"
        )
    
    # ====== GENERATE BUTTON ======
    st.markdown("")  # Add spacing
    col_gen, col_space = st.columns([3, 1])
    
    with col_gen:
        generate_btn = st.button(
            "🚀 Generate Image",
            use_container_width=True,
            type="primary"
        )
    
    # ====== DISPLAY RESULTS ======
    if generate_btn:
        st.info("⏳ Generating image... This may take 30-60 seconds")
        
        image = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed
        )
        
        if image:
            # Save to history
            filepath = save_image(image, prompt)
            st.session_state.generated_images.append(image)
            st.session_state.prompts_history.append(prompt)
            
            # Keep only last 10
            if len(st.session_state.generated_images) > 10:
                st.session_state.generated_images = st.session_state.generated_images[-10:]
                st.session_state.prompts_history = st.session_state.prompts_history[-10:]
            
            # Display image
            st.success("✅ Image generated successfully!")
            st.image(image, caption="Your Generated Artwork", use_column_width=True)
            
            # Download button
            image.save("temp_image.png")
            with open("temp_image.png", "rb") as f:
                st.download_button(
                    label="⬇️ Download Image (PNG)",
                    data=f.read(),
                    file_name=f"ai_art_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )

# ============================================
# SIDEBAR - TIPS & HISTORY
# ============================================
with col_sidebar:
    
    # ====== PROMPT IDEAS ======
    st.subheader("💡 Prompt Ideas")
    
    prompt_categories = {
        "🐉 Fantasy": [
            "Dragon flying over mountains, epic, detailed",
            "Enchanted forest with magical creatures",
            "Castle in clouds, sunset"
        ],
        "🌌 Sci-Fi": [
            "Cyberpunk city, neon lights, rain",
            "Alien landscape, otherworldly",
            "Space station, futuristic"
        ],
        "🎨 Art Styles": [
            "Oil painting, Van Gogh style",
            "Digital art, anime, detailed",
            "Photography, 4K, professional"
        ],
        "🌅 Nature": [
            "Aurora borealis, snowy landscape",
            "Sunset over ocean, golden hour",
            "Mountain landscape, misty, serene"
        ]
    }
    
    selected_category = st.selectbox(
        "Browse Ideas:",
        list(prompt_categories.keys()),
        label_visibility="collapsed"
    )
    
    for idea in prompt_categories[selected_category]:
        if st.button(idea, use_container_width=True, key=idea):
            st.session_state.current_prompt = idea
            st.rerun()
    
    st.divider()
    
    # ====== GENERATION HISTORY ======
    st.subheader("📜 Recent Prompts")
    
    if st.session_state.prompts_history:
        for i, p in enumerate(reversed(st.session_state.prompts_history[-5:]), 1):
            with st.expander(f"#{len(st.session_state.prompts_history) - i + 1}: {p[:40]}..."):
                st.caption(p)
                if st.button("Use this prompt", key=f"use_{i}", use_container_width=True):
                    st.session_state.selected_prompt = p
    else:
        st.info("📝 No generations yet. Create your first image!")

# ============================================
# FOOTER
# ============================================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.metric("Models Used", "Stable Diffusion v1.5")

with footer_col2:
    st.metric("GPU Optimized", "Yes ✅")

with footer_col3:
    st.metric("Open Source", "MIT License")

st.caption("🎨 AI Art Generator | Powered by Stable Diffusion & Streamlit | Made with ❤️")
