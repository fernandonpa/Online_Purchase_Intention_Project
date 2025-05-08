import matplotlib.pyplot as plt
import pandas as pd

class DescriptivePlotter:
    def __init__(self, df):
        self.df = df

    def plot_gender_distribution(self):
        if 'gender_encoded' not in self.df.columns:
            print("Error: 'gender_encoded' column not found.")
            return

        counts = self.df['gender_encoded'].value_counts().sort_index()
        labels = ["Male", "Female", "Prefer not to Say"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color='skyblue')
        ax.set_title("Gender of the Sample")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Persons")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height, int(height),
                    ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_age_distribution(self):
        if 'age_encoded' not in self.df.columns:
            print("Error: 'age_encoded' column not found.")
            return

        counts = self.df['age_encoded'].value_counts().sort_index()
        labels = ["18-25", "25-35", "35-45", "45-55"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values, color='skyblue')
        ax.set_title("Ages of the Sample")
        ax.set_xlabel("Age Range")
        ax.set_ylabel("Number of Persons")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(labels)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height, int(height),
                    ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_age_gender_stacked(self):
        if not {'gender_encoded', 'age_encoded'}.issubset(self.df.columns):
            print("Error: Required columns not found.")
            return

        age_gender_counts = self.df.groupby(['age_encoded', 'gender_encoded']).size().unstack(fill_value=0)
        age_labels = ["18-25", "25-35", "35-45", "45-55"]
        gender_labels = ["Male", "Female", "Prefer not to Say"]
        colors = ['skyblue', 'orchid', 'lightgreen']

        age_gender_counts.index = age_labels
        age_gender_counts.columns = gender_labels

        age_gender_counts.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors)
        plt.title("Age and Gender Distribution")
        plt.xlabel("Age Range")
        plt.ylabel("Number of Persons")
        plt.xticks(rotation=0)
        plt.legend(title="Gender")
        plt.tight_layout()
        plt.show()

    def plot_occupation_distribution(self, prefix='prof_'):
        job_cols = [col for col in self.df.columns if col.startswith(prefix)]
        if not job_cols:
            print(f"No columns with prefix '{prefix}' found.")
            return

        counts = self.df[job_cols].sum().sort_values(ascending=False)
        total = len(self.df)
        percentages = (counts / total * 100).round(2)

        clean_labels = [col.replace(prefix, '').replace('_', ' ').capitalize() for col in counts.index]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(clean_labels, counts.values, color='teal')
        ax.set_title("Occupation Distribution")
        ax.set_ylabel("Number of Respondents")
        ax.set_xticklabels(clean_labels, rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height, int(height),
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        print("\nOccupation Counts and Percentages:")
        for label, count, pct in zip(clean_labels, counts, percentages):
            print(f"{label}: {int(count)} ({pct}%)")

    def plot_multilabel_platform(self, prefix, title):
        platform_cols = [col for col in self.df.columns if col.startswith(prefix)]
        if not platform_cols:
            print(f"No columns with prefix '{prefix}' found.")
            return

        counts = self.df[platform_cols].sum().sort_values(ascending=False)
        total_rows = len(self.df)
        percentages = (counts / total_rows * 100).round(2)

        def clean_name(col):
            return "Not Mentioned" if col.strip() == prefix else col.replace(prefix, '').replace('_', ' ').capitalize()

        clean_labels = [clean_name(col) for col in counts.index]

        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(clean_labels, counts.values, color='darkcyan')
        ax.set_title(title)
        ax.set_ylabel("Number of Respondents")
        ax.set_xticklabels(clean_labels, rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height, int(height),
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        print(f"\n{title} Platform Counts and Percentages:")
        for label, count, pct in zip(clean_labels, counts, percentages):
            print(f"{label}: {int(count)} ({pct}%)")


# plotter = DescriptivePlotter(df)

# plotter.plot_gender_distribution()
# plotter.plot_age_distribution()
# plotter.plot_age_gender_stacked()
# plotter.plot_occupation_distribution()

# plotter.plot_multilabel_platform('gecp_', 'General E-Commerce Platform Usage')
# plotter.plot_multilabel_platform('op_', 'Online Pharmacies Platform Usage')
# plotter.plot_multilabel_platform('fabr_', 'Fashion & Beauty Retailers Platform Usage')

