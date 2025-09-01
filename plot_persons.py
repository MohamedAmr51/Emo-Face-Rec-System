import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button

class ScrollableFaceDisplay:
    def __init__(self, folder_path="Persons_Faces"):
        self.folder_path = folder_path
        self.person_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.person_folders.sort(key=lambda x: int(x.split('_')[1]))
        
        self.rows_per_page = 5  # Number of persons to show at once
        self.current_start = 0
        self.max_photos = 5
        
        self.fig, self.axes = plt.subplots(self.rows_per_page, self.max_photos, figsize=(15, 15))
        self.fig.subplots_adjust(bottom=0.1)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.4, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.5, 0.02, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        
        self.update_display()
        
    def update_display(self):
        # Clear all axes
        for i in range(self.rows_per_page):
            for j in range(self.max_photos):
                self.axes[i, j].clear()
                self.axes[i, j].axis('off')
        
        # Display current page
        end_idx = min(self.current_start + self.rows_per_page, len(self.person_folders))
        
        for row, person_idx in enumerate(range(self.current_start, end_idx)):
            person = self.person_folders[person_idx]
            images = glob.glob(os.path.join(self.folder_path, person, "*.jpg"))
            random.shuffle(images)
            images = images[:self.max_photos]
            
            for col, img_path in enumerate(images):
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                self.axes[row, col].imshow(img)
                self.axes[row, col].axis('off')
            
            # Add person name
            self.axes[row, 0].set_title(person, loc='left')
        
        # Update page info
        total_pages = (len(self.person_folders) + self.rows_per_page - 1) // self.rows_per_page
        current_page = self.current_start // self.rows_per_page + 1
        self.fig.suptitle(f'Page {current_page} of {total_pages}', fontsize=16)
        
        plt.draw()
    
    def prev_page(self, event):
        if self.current_start > 0:
            self.current_start = max(0, self.current_start - self.rows_per_page)
            self.update_display()
    
    def next_page(self, event):
        if self.current_start + self.rows_per_page < len(self.person_folders):
            self.current_start += self.rows_per_page
            self.update_display()
    
    def show(self):
        plt.show()

# Usage
display = ScrollableFaceDisplay("Backup\\DeepFace Management Take 1 outlier removal off , photo size 30x30")
display.show()