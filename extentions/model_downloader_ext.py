import gradio as gr
import os
from modules import config
from modules.model_loader import load_file_from_url
# Removed: from modules.util import get_model_type_list - not found and not strictly needed for this step.

def format_model_type_for_display(model_type_path_or_name: str, is_path=True) -> str:
    """Helper to format model type paths or names for display in dropdown."""
    if is_path:
        # Example: 'models/checkpoints/' -> 'Checkpoints'
        # Example: 'models/loras/' -> 'Loras'
        name = os.path.basename(os.path.normpath(model_type_path_or_name))
        if not name: # If it was a trailing slash like 'models/checkpoints/'
            name = os.path.basename(os.path.dirname(os.path.normpath(model_type_path_or_name)))
    else:
        # Example: 'paths_checkpoints' -> 'Checkpoints'
        name = model_type_path_or_name.replace('path_', '').replace('paths_', '')
    return name.replace('_', ' ').title()

def downloader_ui():
    """Creates the Gradio UI for the model downloader extension."""
    with gr.Blocks() as ui_component:
        gr.Markdown("### Simple Model Downloader\nPaste a direct download URL (e.g., from Hugging Face for a `.safetensors` file) and select the destination model type.")
        with gr.Row():
            model_url = gr.Textbox(label="Model URL", placeholder="Enter Hugging Face model file URL or other direct download link", scale=4)

        model_type_choices = []
        destination_paths_map = {}

        # Checkpoints (potentially multiple paths, use first)
        if hasattr(config, 'paths_checkpoints') and config.paths_checkpoints:
            path = config.paths_checkpoints[0]
            display_name = "Checkpoints" # Explicit name
            model_type_choices.append((display_name, path))
            destination_paths_map[display_name] = path

        # LoRAs (potentially multiple paths, use first)
        if hasattr(config, 'paths_loras') and config.paths_loras:
            path = config.paths_loras[0]
            display_name = "LoRAs" # Explicit name
            model_type_choices.append((display_name, path))
            destination_paths_map[display_name] = path

        # Other single path model types
        single_path_configs = {
            "Embeddings": config.path_embeddings if hasattr(config, 'path_embeddings') else None,
            "VAE": config.path_vae if hasattr(config, 'path_vae') else None,
            "VAE Approx": config.path_vae_approx if hasattr(config, 'path_vae_approx') else None,
            "ControlNet Models": config.path_controlnet if hasattr(config, 'path_controlnet') else None,
            "Upscale Models": config.path_upscale_models if hasattr(config, 'path_upscale_models') else None,
            "Inpaint Models": config.path_inpaint if hasattr(config, 'path_inpaint') else None,
            "CLIP Vision Models": config.path_clip_vision if hasattr(config, 'path_clip_vision') else None,
            "Fooocus Expansion": config.path_fooocus_expansion if hasattr(config, 'path_fooocus_expansion') else None,
        }

        for name, path_val in single_path_configs.items():
            if path_val and isinstance(path_val, str):
                model_type_choices.append((name, path_val))
                destination_paths_map[name] = path_val

        # Fallback if no choices were populated (should not happen in a normal Fooocus setup)
        if not model_type_choices:
            model_type_choices = [("Checkpoints (Fallback)", "models/checkpoints/"), ("LoRAs (Fallback)", "models/loras/")]
            if model_type_choices[0][1] not in destination_paths_map:
                 destination_paths_map[model_type_choices[0][0]] = model_type_choices[0][1]
            if model_type_choices[1][1] not in destination_paths_map:
                 destination_paths_map[model_type_choices[1][0]] = model_type_choices[1][1]


        with gr.Row():
            model_type_dropdown = gr.Dropdown(
                label="Model Type (Destination)",
                choices=[choice[0] for choice in model_type_choices], # Show only display names
                value=model_type_choices[0][0] if model_type_choices else None, # Default to first display name
                scale=2
            )
            download_button = gr.Button("Download Model", scale=1)

        with gr.Row():
            status_text = gr.Textbox(label="Status", value="", interactive=False, lines=3, max_lines=5)

        def handle_download(url, selected_display_name):
            if not url or not selected_display_name:
                return "Error: URL and Model Type must be provided."

            destination_folder = destination_paths_map.get(selected_display_name)

            if not destination_folder:
                return f"Error: Could not determine destination folder for '{selected_display_name}'."

            # Intended status update (Gradio limitation: only final return is typically shown for simple click->textbox)
            # status_text.value = f"Starting download from {url} to {destination_folder}..." # This direct assignment won't work as expected with gr.Button.click

            print(f"[Model Downloader] Attempting to download: {url} to {destination_folder}")

            try:
                # load_file_from_url will create model_dir if it doesn't exist.
                # It also handles progress printing to console.
                # It returns the cached_file path.

                # For UI, it's hard to show real-time progress from torch.hub.download_url_to_file
                # without more complex Gradio queueing/streaming.
                # So, we'll just show a message before and after.
                # A truly dynamic progress bar in the UI would require more work.

                # Placeholder for a "downloading" message if we could update textbox before blocking call
                # For now, this message will be overwritten quickly by return value.
                # yield "Downloading in progress..." # This would require the function to be a generator and click output to handle it.

                file_path = load_file_from_url(url=url, model_dir=destination_folder, progress=True)

                if os.path.exists(file_path):
                    success_message = f"Download complete!\nFile saved to: {file_path}\nDestination folder: {destination_folder}"
                    print(f"[Model Downloader] {success_message}")
                    return success_message
                else:
                    # This case should ideally not be reached if load_file_from_url errors out or returns valid path
                    error_message = f"Error: File not found after download attempt from {url}."
                    print(f"[Model Downloader] {error_message}")
                    return error_message

            except Exception as e:
                error_message = f"Error during download:\n{str(e)}"
                import traceback
                print(f"[Model Downloader] {error_message}\n{traceback.format_exc()}")
                return error_message

        download_button.click(
            handle_download,
            inputs=[model_url, model_type_dropdown],
            outputs=[status_text],
            # _js="() => { document.getElementById('status_textbox_id').value = 'Starting...'; }" # Example of JS update, might need specific elem_id
        )
    return ui_component

if __name__ == '__main__':
    # This is for testing the UI standalone if needed
    # In practice, Fooocus will import and use downloader_ui()

    # Mockup for modules.config for standalone testing
    class MockConfig:
        paths_checkpoints = ["models/checkpoints/"]
        paths_loras = ["models/loras/"]
        path_vae = "models/vae/"
        path_embeddings = "models/embeddings/"
        path_controlnet = "models/controlnet/"
        path_upscale_models = "models/upscale_models/"
        path_inpaint = "models/inpaint/"
        path_clip_vision = "models/clip_vision/"
        path_fooocus_expansion = "models/prompt_expansion/fooocus_expansion/"
        path_vae_approx = "models/vae_approx/"


    # Replace actual config with mock for testing
    original_config = config
    config = MockConfig()

    iface = downloader_ui()
    iface.launch()

    config = original_config # Restore original config

    # To integrate into Fooocus, webui.py will call downloader_ui()
    # Example in webui.py:
    # from extentions import model_downloader_ext # Corrected spelling
    # ...
    # with gr.Accordion('Extention', open=False): # Corrected spelling based on webui.py
    #   with gr.TabItem(label='Model Downloader'):
    #     model_downloader_ext.downloader_ui()
    pass
