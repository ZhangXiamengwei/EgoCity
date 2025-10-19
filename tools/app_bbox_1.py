import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import json

class PointAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Annotation Tool")

        # Data structures
        self.image_paths = []            # List of image file paths
        self.current_image_index = -1    # Which image we're on
        self.original_img = None         # The original unscaled PIL image
        self.img = None                  # Current working PIL image (scaled, if needed)
        self.tkimg = None               # ImageTk object for drawing on Canvas
        self.current_image_path = None   # Path to the currently displayed image
        # We'll store the annotations in this format:
        # {
        #   filename: {
        #       "street_type": "",
        #       "facilities": "",
        #       "points": {
        #           point_id: {
        #               "coords": [x, y],
        #               "type": "",
        #               "activities": [],
        #               "description": ""
        #           },
        #           ...
        #       }
        #   }
        # }
        self.annotations = {}
        self.selected_point_id = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0

        # Track window resizing
        self.root.bind("<Configure>", self.on_window_resize)

        # Keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.save_and_next())
        self.root.bind('<Control-d>', lambda e: self.delete_point())
        # Ctrl+Q now does a silent save (no file dialog):
        self.root.bind('<Control-q>', lambda e: self.save_annotations_silent())
        self.root.bind('<Control-z>', lambda e: self.prev_image())
        self.root.bind('<Control-x>', lambda e: self.next_image())

        self.setup_gui()

    def setup_gui(self):
        # Top frame
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        btn_load = tk.Button(top_frame, text="Load Folder", command=self.load_folder)
        btn_load.pack(side=tk.LEFT, padx=5)

        # Main frame (canvas on left, right panel for points)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(main_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right panel
        right_panel = tk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Image-level fields (street_type, facilities)
        tk.Label(right_panel, text="Image-Level Fields:").pack(anchor=tk.W)
        img_form_frame = tk.Frame(right_panel)
        img_form_frame.pack(fill=tk.X, pady=5)

        tk.Label(img_form_frame, text="Street Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.img_street_type_entry = tk.Entry(img_form_frame, width=20)
        self.img_street_type_entry.grid(row=0, column=1, padx=5, pady=2)
        self.img_street_type_entry.bind('<KeyRelease>', lambda e: self.auto_save_image_info())

        tk.Label(img_form_frame, text="Facilities:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.img_facilities_entry = tk.Entry(img_form_frame, width=20)
        self.img_facilities_entry.grid(row=1, column=1, padx=5, pady=2)
        self.img_facilities_entry.bind('<KeyRelease>', lambda e: self.auto_save_image_info())

        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Point-level fields
        tk.Label(right_panel, text="Points:").pack(anchor=tk.W)
        self.point_listbox = tk.Listbox(right_panel, width=35, height=10)
        self.point_listbox.pack(fill=tk.X, pady=5)
        self.point_listbox.bind("<<ListboxSelect>>",
                                lambda e: self.root.after_idle(
                                    lambda: self.on_point_select(e)))

        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        form_frame = tk.Frame(right_panel)
        form_frame.pack(fill=tk.X, pady=5)

        tk.Label(form_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.type_entry = tk.Entry(form_frame, width=20)
        self.type_entry.grid(row=0, column=1, padx=5, pady=2)
        self.type_entry.bind('<KeyRelease>', lambda e: self.auto_save_point_info())

        tk.Label(form_frame, text="Activities:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.activities_entry = tk.Entry(form_frame, width=20)
        self.activities_entry.grid(row=1, column=1, padx=5, pady=2)
        self.activities_entry.bind('<KeyRelease>', lambda e: self.auto_save_point_info())

        tk.Label(form_frame, text="Description:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.desc_text = tk.Text(form_frame, width=20, height=5)
        self.desc_text.grid(row=2, column=1, padx=5, pady=2)
        self.desc_text.bind('<KeyRelease>', lambda e: self.auto_save_point_info())

        # Return key navigation among point-level fields
        self.type_entry.bind("<Return>", lambda e: self.activities_entry.focus())
        self.activities_entry.bind("<Return>", lambda e: self.desc_text.focus())

        delete_btn = tk.Button(right_panel, text="Delete Point", command=self.delete_point)
        delete_btn.pack(pady=5, fill=tk.X)

        save_btn = tk.Button(right_panel, text="Save Annotations", command=self.save_annotations)
        save_btn.pack(pady=5, fill=tk.X)

        # Bottom nav
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        btn_prev = tk.Button(nav_frame, text="<< Prev", command=self.prev_image)
        btn_prev.pack(side=tk.LEFT, padx=5)
        self.img_label = tk.Label(nav_frame, text="No images loaded.")
        self.img_label.pack(side=tk.LEFT, padx=10)
        btn_next = tk.Button(nav_frame, text="Next >>", command=self.next_image)
        btn_next.pack(side=tk.LEFT, padx=5)

        # Give initial focus to the first text box
        self.type_entry.focus_set()

    # ----------------- Image Loading / Display ------------------
    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        all_files = os.listdir(folder_path)
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in all_files if f.lower().endswith(exts)
        ]
        self.image_paths.sort()

        if not self.image_paths:
            messagebox.showerror("Error", "No valid image files found in this folder.")
            return

        self.current_image_index = 0
        self.load_image()
        self.refresh_point_list()
        self.load_image_level_fields()

    def on_window_resize(self, event):
        if event.widget == self.root:
            self.load_image()

    def load_image(self):
        if not self.image_paths:
            return
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_paths):
            return

        img_path = self.image_paths[self.current_image_index]
        self.img_label.config(text=os.path.basename(img_path))

        # Load new image only if needed
        if not self.original_img or self.current_image_path != img_path:
            self.original_img = Image.open(img_path)
            self.current_image_path = img_path

        # Force canvas update and get dimensions
        self.canvas.update_idletasks()
        cwidth = self.canvas.winfo_width()
        cheight = self.canvas.winfo_height()

        if cwidth <= 1:
            cwidth = self.root.winfo_width() - 300
        if cheight <= 1:
            cheight = self.root.winfo_height() - 100

        ratio = min(
            cwidth / self.original_img.width,
            cheight / self.original_img.height
        )
        new_w = int(self.original_img.width * ratio)
        new_h = int(self.original_img.height * ratio)
        resized_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(resized_img)
        self.scale_factor = ratio
        self.x_offset = (cwidth - new_w) // 2
        self.y_offset = (cheight - new_h) // 2

        self.canvas.delete("all")
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW,
                                 image=self.tkimg, tags="image")
        self.draw_existing_points()
        self.canvas.update_idletasks()

    # ----------------- Optimized Drawing ------------------
    def draw_existing_points(self):
        self.canvas.delete("point_indicator")
        if not self.image_paths:
            return
        filename = os.path.basename(self.image_paths[self.current_image_index])
        if filename not in self.annotations:
            return
        if "points" not in self.annotations[filename]:
            return

        points_coords = []
        text_coords = []

        for point_id, point_data in self.annotations[filename]["points"].items():
            x, y = point_data["coords"]
            cx = self.x_offset + (x * self.scale_factor)
            cy = self.y_offset + (y * self.scale_factor)
            r = 4
            points_coords.extend([
                cx - r, cy - r, cx + r, cy + r,
                "red", "yellow", 1,
                ("point_indicator", f"point_{point_id}")
            ])
            text_coords.extend([
                cx + 5, cy - 10,
                point_id, "yellow", "nw",
                ("point_indicator", f"point_label_{point_id}")
            ])

        for i in range(0, len(points_coords), 8):
            self.canvas.create_oval(
                points_coords[i], points_coords[i + 1],
                points_coords[i + 2], points_coords[i + 3],
                fill=points_coords[i + 4],
                outline=points_coords[i + 5],
                width=points_coords[i + 6],
                tags=points_coords[i + 7]
            )
        for i in range(0, len(text_coords), 6):
            self.canvas.create_text(
                text_coords[i], text_coords[i + 1],
                text=text_coords[i + 2],
                fill=text_coords[i + 3],
                anchor=text_coords[i + 4],
                tags=text_coords[i + 5]
            )

    def draw_point(self, point_id, x, y):
        cx = self.x_offset + (x * self.scale_factor)
        cy = self.y_offset + (y * self.scale_factor)
        r = 4
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="red", outline="yellow", width=1,
            tags=("point_indicator", f"point_{point_id}")
        )
        self.canvas.create_text(
            cx + 5, cy - 10,
            text=point_id, fill="yellow", anchor="nw",
            tags=("point_indicator", f"point_label_{point_id}")
        )

    # ----------------- Navigation ------------------
    def next_image(self):
        if not self.image_paths:
            return
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.original_img = None
            self.tkimg = None
            self.current_image_path = None

            self.canvas.update_idletasks()
            self.root.update_idletasks()

            self.load_image()
            self.canvas.update_idletasks()
            self.root.update_idletasks()

            self.refresh_point_list()
            self.load_image_level_fields()
            self.clear_point_form_fields()

    def prev_image(self):
        if not self.image_paths:
            return
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.original_img = None
            self.tkimg = None
            self.current_image_path = None

            self.canvas.update_idletasks()
            self.root.update_idletasks()

            self.load_image()
            self.canvas.update_idletasks()
            self.root.update_idletasks()

            self.refresh_point_list()
            self.load_image_level_fields()
            self.clear_point_form_fields()

    # ----------------- Point Placement ------------------
    def on_canvas_click(self, event):
        if not self.original_img:
            return

        x = (event.x - self.x_offset) / self.scale_factor
        y = (event.y - self.y_offset) / self.scale_factor

        if (0 <= x <= self.original_img.width and 0 <= y <= self.original_img.height):
            filename = os.path.basename(self.image_paths[self.current_image_index])
            if filename not in self.annotations:
                self.annotations[filename] = {
                    "street_type": "",
                    "facilities": "",
                    "points": {}
                }
            if "points" not in self.annotations[filename]:
                self.annotations[filename]["points"] = {}

            new_id = len(self.annotations[filename]["points"]) + 1
            while str(new_id) in self.annotations[filename]["points"]:
                new_id += 1

            new_id_str = str(new_id)
            self.annotations[filename]["points"][new_id_str] = {
                "coords": [x, y],
                "type": "",
                "activities": [],
                "description": ""
            }

            self.draw_point(new_id_str, x, y)
            self.point_listbox.insert(tk.END, f"Point #{new_id_str}")

            last_idx = self.point_listbox.size() - 1
            self.point_listbox.selection_clear(0, tk.END)
            self.point_listbox.selection_set(last_idx)
            self.point_listbox.activate(last_idx)

            self.selected_point_id = new_id_str
            self.set_form_fields_from_point(filename, new_id_str)

    # ----------------- Point Selection & Editing ------------------
    def refresh_point_list(self):
        if not self.image_paths:
            return
        filename = os.path.basename(self.image_paths[self.current_image_index])
        current_selection = self.point_listbox.curselection()
        selected_index = current_selection[0] if current_selection else None

        self.point_listbox.delete(0, tk.END)
        if filename in self.annotations and "points" in self.annotations[filename]:
            sorted_ids = sorted(self.annotations[filename]["points"].keys(), key=lambda x: int(x))
            self.point_listbox.insert(tk.END, *[f"Point #{pid}" for pid in sorted_ids])
            if selected_index is not None and selected_index < len(sorted_ids):
                self.point_listbox.selection_set(selected_index)

    def on_point_select(self, event):
        if self.selected_point_id:
            self.auto_save_point_info()

        selection = self.point_listbox.curselection()
        if not selection:
            return
        item_text = self.point_listbox.get(selection[0])
        point_id_str = item_text.split("#")[-1].strip()
        if self.selected_point_id != point_id_str:
            self.selected_point_id = point_id_str
            filename = os.path.basename(self.image_paths[self.current_image_index])
            self.set_form_fields_from_point(filename, point_id_str)

    def set_form_fields_from_point(self, filename, point_id_str):
        if filename not in self.annotations:
            return
        if "points" not in self.annotations[filename]:
            return
        if point_id_str not in self.annotations[filename]["points"]:
            return

        point_data = self.annotations[filename]["points"][point_id_str]
        self.root.after_idle(lambda: self._update_form_fields(point_data))

    def _update_form_fields(self, point_data):
        self.type_entry.delete(0, tk.END)
        self.type_entry.insert(0, point_data.get("type", ""))

        self.activities_entry.delete(0, tk.END)
        self.activities_entry.insert(0, ", ".join(point_data.get("activities", [])))

        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END, point_data.get("description", ""))

    def clear_point_form_fields(self):
        self.selected_point_id = None
        self.type_entry.delete(0, tk.END)
        self.activities_entry.delete(0, tk.END)
        self.desc_text.delete("1.0", tk.END)

    def auto_save_point_info(self):
        if self.selected_point_id is None:
            return
        filename = os.path.basename(self.image_paths[self.current_image_index])
        if (filename not in self.annotations or
                "points" not in self.annotations[filename] or
                self.selected_point_id not in self.annotations[filename]["points"]):
            return

        self.annotations[filename]["points"][self.selected_point_id].update({
            "type": self.type_entry.get().strip(),
            "activities": [a.strip() for a in self.activities_entry.get().split(",") if a.strip()],
            "description": self.desc_text.get("1.0", tk.END).strip()
        })

    def delete_point(self):
        if self.selected_point_id is None:
            return
        filename = os.path.basename(self.image_paths[self.current_image_index])
        if (filename in self.annotations
                and "points" in self.annotations[filename]
                and self.selected_point_id in self.annotations[filename]["points"]):
            del self.annotations[filename]["points"][self.selected_point_id]

        self.selected_point_id = None
        self.clear_point_form_fields()

        self.canvas.delete("point_indicator")
        self.draw_existing_points()
        self.refresh_point_list()

    # ----------------- Image-Level Editing ------------------
    def load_image_level_fields(self):
        filename = os.path.basename(self.image_paths[self.current_image_index])
        if filename not in self.annotations:
            # Initialize if not present
            self.annotations[filename] = {
                "street_type": "",
                "facilities": "",
                "points": {}
            }
        self.img_street_type_entry.delete(0, tk.END)
        self.img_street_type_entry.insert(0, self.annotations[filename].get("street_type", ""))
        self.img_facilities_entry.delete(0, tk.END)
        self.img_facilities_entry.insert(0, self.annotations[filename].get("facilities", ""))

    def auto_save_image_info(self):
        filename = os.path.basename(self.image_paths[self.current_image_index])
        if filename not in self.annotations:
            self.annotations[filename] = {
                "street_type": "",
                "facilities": "",
                "points": {}
            }
        self.annotations[filename]["street_type"] = self.img_street_type_entry.get().strip()
        self.annotations[filename]["facilities"] = self.img_facilities_entry.get().strip()

    # ----------------- Saving ------------------
    def save_annotations(self):
        # Auto-save current point before saving to file
        if self.selected_point_id:
            self.auto_save_point_info()

        if not self.annotations:
            messagebox.showinfo("No Annotations", "No annotations to save yet.")
            return

        # Default path based on current image
        current_image = self.image_paths[self.current_image_index]
        default_path = os.path.splitext(current_image)[0] + '.json'

        file_path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(default_path),
            initialdir=os.path.dirname(default_path),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Saved", f"Annotations saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def save_annotations_silent(self):
        # Auto-save current point before saving
        if self.selected_point_id:
            self.auto_save_point_info()
        if not self.annotations:
            return  # do nothing silently

        current_image = self.image_paths[self.current_image_index]
        default_path = os.path.splitext(current_image)[0] + '.json'

        try:
            with open(default_path, "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, indent=4, ensure_ascii=False)
        except Exception:
            pass  # even if error, remain silent

    def save_and_next(self):
        if not self.point_listbox.size():
            return

        current_selection = self.point_listbox.curselection()
        if not current_selection:
            next_index = 0
        else:
            next_index = current_selection[0] + 1

        if next_index >= self.point_listbox.size():
            next_index = 0

        self.point_listbox.selection_clear(0, tk.END)
        self.point_listbox.selection_set(next_index)
        self.point_listbox.activate(next_index)
        self.point_listbox.see(next_index)
        self.point_listbox.event_generate('<<ListboxSelect>>')

if __name__ == "__main__":
    root = tk.Tk()
    app = PointAnnotator(root)
    root.mainloop()
