import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from classifier import train_model

model, scaler, train_acc, test_acc, k_scores, cm, report = train_model()

best_k = 4
MAX_STUDY_HOURS = 80

def predict():
    try:
        attendance = float(entry_attendance.get())
        marks = float(entry_marks.get())
        assignments = float(entry_assignments.get())
        hours = float(entry_hours.get())
        backlogs = float(entry_backlogs.get())

        if any(v < 0 for v in [attendance, marks, assignments, hours, backlogs]):
            output_label.config(text="Values cannot be negative", fg="orange")
            root.update()
            return
        if attendance > 100:
            output_label.config(text="Attendance cannot exceed 100%", fg="orange")
            root.update()
            return
        if marks > 100:
            output_label.config(text="Marks cannot exceed 100", fg="orange")
            root.update()
            return
        if assignments > 100:
            output_label.config(text="Assignments cannot exceed 100%", fg="orange")
            root.update()
            return
        if hours > MAX_STUDY_HOURS:
            output_label.config(text=f"Study hours cannot exceed {MAX_STUDY_HOURS}hrs/week", fg="orange")
            root.update()
            return

        features = np.array([[attendance, marks, assignments, hours, backlogs]])
        scaled = scaler.transform(features)
        result = model.predict(scaled)
        output_label.config(text="Result: PASS ✓" if result[0]==1 else "Result: FAIL ✗",
                           fg="green" if result[0]==1 else "red")
        root.update()

    except ValueError:
        output_label.config(text="Please enter valid numbers", fg="orange")
        root.update()

def reset():
    entry_attendance.delete(0, tk.END)
    entry_marks.delete(0, tk.END)
    entry_assignments.delete(0, tk.END)
    entry_hours.delete(0, tk.END)
    entry_backlogs.delete(0, tk.END)
    output_label.config(text="")
    root.update()

def open_analysis():
    win = tk.Toplevel(root)
    win.title("Model Analysis — KNN")
    win.geometry("750x800")

    # K Table — horizontal
    tk.Label(win, text="K Value vs Accuracy", font=("Arial", 12, "bold")).pack(pady=8)
    table_frame = tk.Frame(win)
    table_frame.pack()

    headers = ["k"] + [str(k) for k in k_scores.keys()]
    accs = ["Acc %"] + [str(a) for a in k_scores.values()]

    for col, val in enumerate(headers):
        is_best = (col > 0 and list(k_scores.keys())[col-1] == best_k)
        bg = "#90EE90" if is_best else "#333333"
        tk.Label(table_frame, text=val, width=7, font=("Arial", 9, "bold"),
                 bg=bg, fg="white", relief="ridge", pady=4).grid(row=0, column=col)

    for col, val in enumerate(accs):
        is_best = (col > 0 and list(k_scores.keys())[col-1] == best_k)
        bg = "#90EE90" if is_best else "white"
        fg = "black"
        tk.Label(table_frame, text=val, width=7, font=("Arial", 9),
                 bg=bg, fg=fg, relief="ridge", pady=4).grid(row=1, column=col)

    tk.Label(win, text=f"★ k={best_k} highlighted in green",
             font=("Arial", 9), fg="gray").pack(pady=3)

    # Confusion Matrix
    tk.Label(win, text="Confusion Matrix", font=("Arial", 12, "bold")).pack(pady=8)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (k={best_k})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Metrics Table
    tk.Label(win, text="Precision / Recall / F1 Score", font=("Arial", 12, "bold")).pack(pady=8)
    metrics_frame = tk.Frame(win)
    metrics_frame.pack()

    headers = ["Class", "Precision", "Recall", "F1 Score", "Support"]
    for col, h in enumerate(headers):
        tk.Label(metrics_frame, text=h, width=12, font=("Arial", 9, "bold"),
                 bg="#333333", fg="white", relief="ridge", pady=4).grid(row=0, column=col)

    rows = [
        ["Fail", "1.00", "0.33", "0.50", "9"],
        ["Pass", "0.83", "1.00", "0.91", "29"],
        ["Macro Avg", "0.91", "0.67", "0.70", "38"],
    ]
    colors = ["#FFEEEE", "#EEFFEE", "#F5F5F5"]
    for r, (row, color) in enumerate(zip(rows, colors), start=1):
        for col, val in enumerate(row):
            tk.Label(metrics_frame, text=val, width=12, font=("Arial", 9),
                     bg=color, relief="ridge", pady=4).grid(row=r, column=col)

    tk.Label(win, text=f"Train Accuracy: {train_acc*100:.1f}%     Test Accuracy: {test_acc*100:.1f}%",
             font=("Arial", 10, "bold"), fg="#333333").pack(pady=12)

# Main window
root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("500x560")

tk.Label(root, text="Student Performance Predictor", font=("Arial", 14, "bold")).pack(pady=10)
tk.Label(root, text=f"Model: KNN  |  Best k={best_k}  |  Train Acc: {train_acc*100:.1f}%  |  Test Acc: {test_acc*100:.1f}%",
         font=("Arial", 9), fg="gray").pack(pady=2)

tk.Label(root, text="Attendance %").pack()
entry_attendance = tk.Entry(root)
entry_attendance.pack()

tk.Label(root, text="Average Marks").pack()
entry_marks = tk.Entry(root)
entry_marks.pack()

tk.Label(root, text="Assignments Completed %").pack()
entry_assignments = tk.Entry(root)
entry_assignments.pack()

tk.Label(root, text=f"Study Hours per Week (max {MAX_STUDY_HOURS})").pack()
entry_hours = tk.Entry(root)
entry_hours.pack()

tk.Label(root, text="Backlogs").pack()
entry_backlogs = tk.Entry(root)
entry_backlogs.pack()

tk.Button(root, text="Predict", command=predict, bg="blue", fg="white").pack(pady=15)
tk.Button(root, text="Reset", command=reset, bg="gray", fg="white").pack(pady=5)
tk.Button(root, text="View Model Analysis", command=open_analysis,
          bg="#333333", fg="white").pack(pady=8)

output_label = tk.Label(root, text="", font=("Arial", 13, "bold"))
output_label.pack()

root.lift()
root.focus_force()
root.mainloop()