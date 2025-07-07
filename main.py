import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from unet import UNet3D, load_checkpoint, segment
from dataset import load_uploaded_volume, filter_modalities, dataset_val
from metrics import dice_score, iou_score, filter_regions


# Path to your processed BraTS data
VOLUME_DIR = "data/processed/input_data_processed/volumes"
MASK_DIR   = "data/processed/input_data_processed/masks"

DEVICE = 'cuda'


# -----------------
# utility functions
# -----------------
@st.cache_resource
def get_model(model_path: str):
    model = UNet3D()
    load_checkpoint(model_path, model)
    model.to(DEVICE)
    return model

@st.cache_data
def cached_segment(input_bytes: bytes, model_path: str):
    model = get_model(model_path)
    volume = np.frombuffer(input_bytes, dtype=np.float32).reshape(3, 128, 128, 128)
    return segment(volume, model, device=DEVICE)


# -----------------
# Main Program
# -----------------

st.set_page_config(layout="centered")
st.title("3D UNet Segmentation Explorer")

# Initialize segmented state
if 'segmented' not in st.session_state:
    st.session_state.segmented = False


# --- Model Selection ---
source_model = st.radio(
    "Model source", ["Default models", "Upload your own"]
)
if source_model == "Default models":
    saved = sorted(os.listdir("saved_models"))
    chosen = st.selectbox("Select a model", saved)
    model_path = os.path.join("saved_models", chosen)
    uploaded_model = None
else:
    uploaded_model = st.file_uploader(
        "Upload a model checkpoint (.pt)", type=["pt"]
    )

    if uploaded_model is not None:
        model_path = "uploaded_model.pt"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
    else:
        st.warning("Please upload a model file to continue.")
        st.stop()

# store previous choice to detect changes | resets the prediction button every time
# if 'last_model_path' not in st.session_state:
#     st.session_state.last_model_path = None

# current_model_id = model_path or (uploaded_model.name if uploaded_model else None)
# if current_model_id != st.session_state.last_model_path:
#     st.session_state.segmented = False
#     st.session_state.last_model_path = current_model_id





# --- Average Metrics Button ---

# Define the callback
def start_metrics():
    st.session_state.metrics_started = True

# Initialize the state
if "metrics_started" not in st.session_state:
    st.session_state.metrics_started = False

# Show "Calculate" button if not started
if not st.session_state.metrics_started:
    st.button("Calculate average metrics", on_click=start_metrics)


if st.session_state.metrics_started:
    metric_region = st.selectbox(
    "Select clinical region to evaluate",
    ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    )

    if st.button("Proceed"):
        with st.spinner("Evaluating on validation set..."):
            # initialize progress bar
            total = len(dataset_val)
            progress = st.progress(0)

            dices, ious = [], []
            for idx, (vol, mask) in enumerate(dataset_val):
             
                vol_np = vol.cpu().numpy()
                mask_np = mask.cpu().numpy()
                mask_np = filter_regions(mask_np, metric_region)
     
                    
                input_bytes = vol_np.tobytes()
                pred = cached_segment(input_bytes, model_path)
                pred = filter_regions(pred, metric_region)

                dices.append(dice_score(pred, mask_np))
                ious.append(iou_score(pred, mask_np))

                # update progress
                progress.progress((idx + 1) / total)

            avg_dice = np.mean(dices)
            avg_iou = np.mean(ious)
        st.success(
            f"Average Dice: {avg_dice:.4f} | Average IoU: {avg_iou:.4f}"
        )


if st.session_state.metrics_started:
    st.stop()

# --- Data Source Selection ---
source = st.radio("Data source", ["Use BraTS dataset", "Upload your own"])

if source == "Upload your own":
    uploaded = st.file_uploader("Upload an MRI volume (.npy or .nii)", type=["npy","nii"])
    if uploaded:
        volume = load_uploaded_volume(uploaded)
        channel_names = [f"channel {i+1}" for i in range(volume.shape[0])]
        default_channel_names = [f"channel 1", "channel 2", "channel 3"]
        custom_source = True
        mask = None

else:
    all_vols = sorted(os.listdir(VOLUME_DIR))
    choice   = st.selectbox("Select an MRI volume", all_vols)
    volume = np.load(os.path.join(VOLUME_DIR, choice))
    raw_mask  = np.load(os.path.join(MASK_DIR, choice)).astype(np.uint8)
    mask = np.argmax(raw_mask, axis=-1).astype(np.int64)
    channel_names = ['flair','t1','t1ce','t2']
    default_channel_names = ['flair', 't1ce', 't2']

# Proceed if volume loaded
if 'volume' in locals():
    C, D, H, W = volume.shape

    slice_idx   = st.slider("Slice index (depth)", 0, D-1, D//2)

    if C != 1:
        channel_idx = st.slider("Channel index", 0, C-1, 0)

        selected = st.multiselect(
            "Select 3 input modalities for prediction (in order flair, t1, t1ce)",
            channel_names,
            default=default_channel_names,
            max_selections=3
        )

        if len(selected) == 2:
            st.warning(
                "Choosing only two modalities may causes errors as the model is built for 1 or 3. Selecting exactly 1 or 3 channels will give better predictions"
            )
    else:
        channel_idx = 0

    if C == 3:
        input = filter_modalities(volume, modalities=channel_names, selected=selected)
    else:
        input = np.repeat(volume[0:1], 3, axis=0)

    input_bytes = input.astype(np.float32).tobytes()
    
    # # plotting using pyplot
    # fig5, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(mask[:, :, slice_idx], cmap='jet')
    # axes[0].set_title("Ground Truth")
    # axes[0].axis('off')
    # axes[1].imshow(pred[:, :, slice_idx], cmap='jet')
    # axes[1].set_title("Prediction")
    # axes[1].axis('off')
    # fig5.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05)
    # st.pyplot(fig5)


    st.markdown("---")
    col1, col2 = st.columns(2)

    # --- LEFT: Input & GT heatmap (if applicable) ---
    with col1:
        st.markdown(f"**Input: {channel_names[channel_idx]} (Slice {slice_idx})**")

        # placeholder for chart
        chart_ph = st.empty()

        # checkbox to show gt image
        show_gt = mask is not None and st.checkbox("Show ground truth overlay")

        # prepare the base image
        img_slice = volume[channel_idx, :, :, slice_idx]
        norm      = (img_slice - img_slice.min()) / (np.ptp(img_slice) + 1e-8)
        rgb       = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)

        # build the figure
        fig = go.Figure()
        fig.add_trace(go.Image(z=rgb, hoverinfo='skip'))

        # add the GT overlay if chekced
        if show_gt:
            gt_slice  = mask[:, :, slice_idx]
            legend   = {0:"Background",1:"Non-Enhancing Tumor",2:"Edema",3:"Enhancing Tumor"}
            htext    = np.vectorize(legend.get)(gt_slice)

            fig.add_trace(go.Heatmap(
                z=gt_slice,
                hovertext=htext,
                hoverinfo="text",
                colorscale=[
                    [0.0, "rgba(0,0,0,0)"],
                    [0.25, "rgba(255,0,0,0.5)"],
                    [0.5, "rgba(255,255,0,0.5)"],
                    [0.75, "rgba(0,255,255,0.5)"],
                    [1.0, "rgba(255,0,255,0.5)"],
                ],
                showscale=False,
                opacity=0.6
            ))
            # st.markdown("""
            #     <style>
            #     .legend-block {
            #         display: flex;
            #         flex-direction: column;
            #         gap: 0.4em;
            #     }
            #     .legend-item {
            #         display: flex;
            #         align-items: center;
            #         gap: 0.5em;
            #         font-size: 0.9em;
            #     }
            #     .color-box {
            #         width: 20px;
            #         height: 20px;
            #         border-radius: 3px;
            #         display: inline-block;
            #     }
            #     </style>

            #     <div class="legend-block">
            #         <div class="legend-item"><div class="color-box" style="background-color: rgba(0,0,0,0.5);"></div> Background</div>
            #         <div class="legend-item"><div class="color-box" style="background-color: rgba(255,0,0,0.5);"></div> Non-Enhancing Tumor</div>
            #         <div class="legend-item"><div class="color-box" style="background-color: rgba(255,255,0,0.5);"></div> Edema</div>
            #         <div class="legend-item"><div class="color-box" style="background-color: rgba(0,255,255,0.5);"></div> Enhancing Tumor</div>
            #     </div>
            #     """, unsafe_allow_html=True)

        fig.update_layout(
            width=350,
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        # render into the placeholder
        chart_ph.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


    # --- RIGHT: Prediction heatmap ---
    with col2:
        st.markdown("**Prediction (heatmap)**")

        result_ph = st.empty()
        button_ph = st.empty()

        # Show Predict button until segmentation is triggered
        if not st.session_state.segmented:
            if button_ph.button("Segment Input"):
                st.session_state.segmented = True
                # force immediate rerun so button hides
                button_ph.empty() 

        if st.session_state.segmented:
            pred = cached_segment(input_bytes, model_path)
            pred_slice = pred[:, :, slice_idx]

            legend_pred = {
                0: "Background", 
                1: "Non-Enhancing Tumor", 
                2: "Edema", 
                3: "Enhancing Tumor"
            }
            htext_pred = np.vectorize(legend_pred.get)(pred_slice)

            # build the figure & add heatmap data
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Heatmap(
            z=pred_slice,
            hovertext=htext_pred,
            hoverinfo="text",
            colorscale=[
                [0.0, "rgba(0,0,0,1)"],
                [0.25, "rgba(255,0,0,0.5)"],
                [0.5, "rgba(255,255,0,0.5)"],
                [0.75, "rgba(0,255,255,0.5)"],
                [1.0, "rgba(255,0,255,0.5)"],
            ],
            showscale=False,
            opacity=1
            ))


            # add the GT overlay if checked
            if show_gt:
                gt_slice  = mask[:, :, slice_idx]
                legend   = {0:"Background",1:"Non-Enhancing Tumor",2:"Edema",3:"Enhancing Tumor"}
                htext    = np.vectorize(legend.get)(gt_slice)

                fig_pred.add_trace(go.Heatmap(
                    z=gt_slice,
                    hovertext=htext,
                    hoverinfo="text",
                    colorscale=[
                    [0.0, "rgba(0,0,0,0)"],
                    [0.25, "rgba(255,0,0,0.5)"],
                    [0.5, "rgba(255,255,0,0.5)"],
                    [0.75, "rgba(0,255,255,0.5)"],
                    [1.0, "rgba(255,0,255,0.5)"],
                    ],
                    showscale=False,
                    opacity=0.6
                ))
            
            fig_pred.update_layout(
                width=350,
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(
                    visible=False,
                    autorange='reversed'
                )
            )

            # render the plot
            st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

            if show_gt:
                # show metrics
                dice = dice_score(pred_slice, mask[:, :, slice_idx])
                iou = iou_score(pred_slice, mask[:, :, slice_idx])
                st.markdown(
                    f"**Metrics (Slice {slice_idx}):**  "
                    f"- Dice `{dice:.4f}`, IoU `{iou:.4f}`"
                )



st.markdown("---")
# st.markdown("### Ground Truth Visualization")

# # Adjust column widths to reduce space and better balance
# overlay_col, legend_col = st.columns([4, 1.2], gap=None)

# with overlay_col:
#     # Prepare base image
#     img_slice = volume[channel_idx, :, :, slice_idx]
#     norm = (img_slice - img_slice.min()) / (np.ptp(img_slice) + 1e-8)
#     rgb = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)

#     # GT mask slice
#     gt_slice = mask[:, :, slice_idx]
#     legend = {0: "Background", 1: "Non-Enhancing Tumor", 2: "Edema", 3: "Enhancing Tumor"}
#     htext = np.vectorize(legend.get)(gt_slice)

#     # Build figure
#     fig_gt = go.Figure()
#     fig_gt.add_trace(go.Image(z=rgb, hoverinfo='skip'))

#     fig_gt.add_trace(go.Heatmap(
#         z=gt_slice,
#         hovertext=htext,
#         hoverinfo="text",
#         colorscale=[
#             [0.0, "rgba(0,0,0,0)"],         # Background
#             [0.2499, "rgba(0,0,0,0)"],
#             [0.25, "rgba(255,0,0,0.5)"],    # Non-Enhancing Tumor
#             [0.3499, "rgba(255,0,0,0.5)"],
#             [0.35, "rgba(0,255,255,0.5)"],  # Edema
#             [0.4499, "rgba(0,255,255,0.5)"],
#             [0.45, "rgba(255,0,255,0.5)"],  # Enhancing Tumor
#             [1.0, "rgba(255,0,255,0.5)"],
#         ],
#         zmin=0,
#         zmax=3,
#         showscale=False,
#         opacity=0.6
#     ))

#     fig_gt.update_layout(
#         width=350,
#         height=350,
#         margin=dict(l=0, r=0, t=0, b=0),
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False)
#     )

#     st.plotly_chart(fig_gt, use_container_width=True, config={"displayModeBar": False})

# with legend_col:
#     # Add a container div with flexbox that centers vertically and aligns left
#     st.markdown("""
#         <style>
#         .legend-container {
#             height: 350px; /* same as image height */
#             display: flex;
#             flex-direction: column;
#             justify-content: center;  /* vertical center */
#             align-items: flex-start;  /* align to left */
#             gap: 0.6em;
#             padding-left: -0.3em; /* small left padding to avoid touching edge */
#             font-size: 0.85em;
#             margin-left: -50px;
#         }
#         .legend-item {
#             display: flex;
#             align-items: center;
#             gap: 0.5em;
#         }
#         .color-box {
#             width: 18px;
#             height: 18px;
#             border-radius: 3px;
#             display: inline-block;
#         }
#         </style>

#         <div class="legend-container">
#             <div class="legend-item"><div class="color-box" style="background-color: rgb(0,0,0);"></div> Background</div>
#             <div class="legend-item"><div class="color-box" style="background-color: rgb(255,0,0);"></div> Non-Enhancing Tumor</div>
#             <div class="legend-item"><div class="color-box" style="background-color: rgb(0,255,255);"></div> Edema</div>
#             <div class="legend-item"><div class="color-box" style="background-color: rgb(255,0,255);"></div> Enhancing Tumor</div>
#         </div>
#     """, unsafe_allow_html=True)
