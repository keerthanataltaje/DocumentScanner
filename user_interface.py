# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:04:33 2021

@author: Keerthana
"""
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


def display(new_path):
    window=tk.Tk()
    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    window.geometry(f'{w}x{h}')
    print(new_path)
    
    load=Image.open(new_path)
    render=ImageTk.PhotoImage(load)
    img_frame=Label(window,image=render)
    img_frame.place(x=24, y=80)
   
    Button(window, text="Quit", command=window.destroy).pack()
    window.mainloop() 



