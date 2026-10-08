# 🍽️ AI Food Recognition & Calorie Estimation

An AI-powered food recognition and calorie estimation system that identifies food items from images and estimates their calorie content based on the predicted food and user-provided portion size.

The project uses a pretrained EfficientNet-B0 model fine-tuned on the Food-101 dataset and provides a simple desktop GUI built with Tkinter.

---

## 📌 Project Overview

The objective of this project is to develop an intelligent system that can:

- 📷 Accept a food image from the user
- 🤖 Identify the food item using Deep Learning
- 📊 Display prediction confidence
- 🏆 Display the top food predictions
- ⚖️ Accept the food portion size in grams
- 🔥 Estimate calories based on the predicted food
- 🖥️ Provide an easy-to-use desktop graphical interface

This project was developed as part of the **Prodigy Infotech Machine Learning Internship – Task 05**.

---

# 🎯 Features

## 1. Food Image Upload

Users can select food images from their computer.

Supported formats:

- JPG
- JPEG
- JFIF
- PNG
- WEBP
- BMP

---

## 2. AI Food Recognition

The system uses:

**EfficientNet-B0**

fine-tuned on the **Food-101 dataset**.

The model can recognize **101 different food categories**.

---

## 3. Confidence Score

The application displays the confidence score of the predicted food.

Example:

```text
Food: Pizza

Confidence: 81.85%
