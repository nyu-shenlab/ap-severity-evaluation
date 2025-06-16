import os
import matplotlib.pyplot as plt

def create_visualizations(gradcam_dir,
                        original_image, 
                        cam_image,
                        rid,
                        phase_idx,
                        img_idx,
                        label):
    rid_dir = os.path.join(gradcam_dir, str(rid))
    os.makedirs(rid_dir, exist_ok=True)

    # Create figure with subplots
    plt.figure(figsize=(10, 4))
    
    # Plot original image
    plt.subplot(121)
    plt.imshow(original_image.transpose(1, 2, 0))  # [H, W, C]
    plt.title(f'Original Image\n{label}')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(122)
    plt.imshow(original_image.transpose(1, 2, 0))
    plt.imshow(cam_image, cmap='jet', alpha=0.5)
    plt.title(f'GradCAM Overlay\nPhase {phase_idx}, Slice {img_idx}')
    plt.axis('off')
    
    # Add a main title
    plt.suptitle(f'Patient Scan - {label}', fontsize=14, y=1.05)
    
    # Save figure
    save_path = os.path.join(rid_dir, f'rid{rid}_phase{phase_idx}_img{img_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()