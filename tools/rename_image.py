import os

def rename_images(folder_path, total_images):
    """
    Rename images in the given folder sequentially from 000.png to (total_images - 1).png.

    :param folder_path: Path to the folder containing the images.
    :param total_images: Total number of images to rename.
    """
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            return

        # List all files in the folder
        files = sorted(os.listdir(folder_path))
        
        # Filter out files that are not images (only jpg, jpeg, png, etc.)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

        if len(image_files) == 0:
            print("No image files found in the folder.")
            return

        # Ensure total_images does not exceed the number of available images
        total_images = min(total_images, len(image_files))

        # Rename images
        for index, filename in enumerate(image_files[:total_images]):
            # Generate the new file name
            new_name = f"{index:04d}.png"

            # Get full paths for renaming
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

        print("Renaming completed!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
folder_path = "F:/destop/video_sample/test_simple_0312"  # Replace with the actual folder path
total_images = 100  # Adjust as needed
rename_images(folder_path, total_images)
