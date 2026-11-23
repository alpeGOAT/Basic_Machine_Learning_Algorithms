import tkinter as tk
from tkinter import ttk

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

BAR_COLOR = 'royalblue'
SCATTER_COLOR = 'red'

# Colors for the tkinter widgets and the matplotlib charts in each mode
THEMES = {
    'light': {
        'bg': '#f0f0f0', 'panel': '#ffffff', 'fg': '#1a1a1a', 'button': '#e1e1e1',
        'button_active': '#d0d0d0', 'figure': '#ffffff', 'axes': '#ffffff', 'grid': '#b0b0b0',
    },
    'dark': {
        'bg': '#1e1f22', 'panel': '#2b2d31', 'fg': '#e6e6e6', 'button': '#3a3c42',
        'button_active': '#4a4d55', 'figure': '#1e1f22', 'axes': '#2b2d31', 'grid': '#55585f',
    },
}


def chart_rc(theme):
    """matplotlib settings that make a chart match the given theme."""
    c = THEMES[theme]
    return {
        'figure.facecolor': c['figure'], 'axes.facecolor': c['axes'],
        'axes.edgecolor': c['fg'], 'axes.labelcolor': c['fg'], 'axes.titlecolor': c['fg'],
        'xtick.color': c['fg'], 'ytick.color': c['fg'], 'text.color': c['fg'],
        'grid.color': c['grid'],
        'boxplot.boxprops.color': c['fg'], 'boxplot.whiskerprops.color': c['fg'],
        'boxplot.capprops.color': c['fg'], 'boxplot.flierprops.markeredgecolor': c['fg'],
    }

airbnb = pd.read_csv('Aemf1.csv')
airbnb['Superhost'] = airbnb['Superhost'].astype('int')


def bar(ax, series, title, xlabel, ylabel, rotate=False):
    ax.bar(series.index, series.values, color=BAR_COLOR, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.5, axis='y')
    if rotate:
        ax.tick_params(axis='x', rotation=45)


def scatter(ax, column, title, xlabel):
    ax.scatter(airbnb[column], airbnb['Price'], color=SCATTER_COLOR, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Price')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.5)


# ---------- chart builders (each draws onto a Figure) ----------

def city_distribution(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['City'].value_counts().sort_index(),
        'City Distribution of Airbnb Dataset', 'City', 'Value Count', rotate=True)


def avg_price_each_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Price'].mean().round(2),
        'Average Price for each City', 'City', 'Average Price', rotate=True)


def day_distribution(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['Day'].value_counts().sort_index(),
        'Day Distribution of Airbnb Dataset', 'Day Type', 'Value Count')
    ax.set_ylim(20800, 20900)
    ax.set_yticks(range(20800, 20901, 10))


def room_type_distribution(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['Room Type'].value_counts().sort_index(),
        'Room Distribution of Airbnb Dataset', 'Room Type', 'Value Count')
    ax.set_yticks([0, 1000, 5000, 10000, 15000, 20000, 25000, 30000])


def avg_capacity_each_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Person Capacity'].mean().round(2),
        'Average Person Capacity of each City', 'City', 'Average Person Capacity', rotate=True)


def host_type_distribution(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['Superhost'].value_counts().sort_index(),
        'Host Type Distribution of Airbnb Dataset', 'Superhost Type', 'Value Count')
    ax.set_xticks([0, 1])


def business_or_not(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['Business'].value_counts().sort_index(),
        'Room Type Distribution (Business or not)', 'Business or not', 'Value Count')
    ax.set_xticks([0, 1])


def multiple_rooms(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb['Multiple Rooms'].value_counts().sort_index(),
        'Does Airbnb has multiple rooms?', 'Room Amount (Multiple or not)', 'Value Count')
    ax.set_xticks([0, 1], ['No', 'Yes'])


def avg_price_by_room_type(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('Room Type')['Price'].mean().round(2),
        'Average Price based on Room Type', 'Room Type', 'Average Price')


def avg_price_by_capacity(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('Person Capacity')['Price'].mean().round(2),
        'Average Price based on Person Capacity', 'Person Capacity', 'Average Price')


def avg_cleanliness_by_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Cleanliness Rating'].mean().round(2),
        'Average Cleanliness Rating by City', 'City', 'Average Cleanliness Rating', rotate=True)


def avg_satisfaction_by_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Guest Satisfaction'].mean().round(2),
        'Average Guest Satisfaction by City', 'City', 'Average Guest Satisfaction', rotate=True)


def most_expensive_by_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Price'].max().round(),
        'Most Expensive Airbnb by City', 'City', 'Most Expensive Airbnb Price', rotate=True)


def satisfaction_by_host_type(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('Superhost')['Guest Satisfaction'].mean().round(2),
        'Average Guest Satisfaction Rate by Host Type', 'Host Type', 'Average Guest Satisfaction Rate')
    ax.set_xticks([0, 1], ['Normal Host', 'Super Host'])


def price_by_host_type(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('Superhost')['Price'].mean().round(2),
        'Average Price based on Host Type', 'Host Type', 'Average Price')
    ax.set_xticks([0, 1], ['Normal Host', 'Super Host'])


def price_vs_city_center(fig):
    scatter(fig.add_subplot(), 'City Center (km)',
            'Price Depending on Distance from the City Center', 'City Center (km)')


def superhost_pct_by_city(fig):
    ax = fig.add_subplot()
    bar(ax, airbnb.groupby('City')['Superhost'].mean().mul(100).round(2),
        'Superhost Percentage by City', 'City', 'Superhost Percentage', rotate=True)


def price_vs_restaurant(fig):
    scatter(fig.add_subplot(), 'Restraunt Index',
            'Price Depending on Restaurant Index', 'Restaurant Index')


def price_vs_attraction(fig):
    scatter(fig.add_subplot(), 'Attraction Index',
            'Price Depending on Attraction Index', 'Attraction Index')


def correlation_matrix(fig):
    ax = fig.add_subplot()
    corr = airbnb.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, annot_kws={'size': 7})
    ax.set_title('Correlation Matrix of Airbnb Dataset', fontsize=14)
    ax.tick_params(labelsize=8)


def price_boxplot(fig):
    ax = fig.add_subplot()
    ax.boxplot(airbnb['Price'])
    ax.set_ylabel('Price')
    ax.set_title('Price Boxplot of Airbnb Dataset', fontsize=14)
    ax.grid(True, alpha=0.5, axis='y')


CHARTS = {
    'City Distribution': city_distribution,
    'Average Price by City': avg_price_each_city,
    'Day Type Distribution': day_distribution,
    'Room Type Distribution': room_type_distribution,
    'Avg Person Capacity by City': avg_capacity_each_city,
    'Host Type Distribution': host_type_distribution,
    'Business or Not': business_or_not,
    'Multiple Rooms or Not': multiple_rooms,
    'Avg Price by Room Type': avg_price_by_room_type,
    'Avg Price by Capacity': avg_price_by_capacity,
    'Avg Cleanliness by City': avg_cleanliness_by_city,
    'Avg Guest Satisfaction by City': avg_satisfaction_by_city,
    'Most Expensive by City': most_expensive_by_city,
    'Satisfaction by Host Type': satisfaction_by_host_type,
    'Avg Price by Host Type': price_by_host_type,
    'Price vs City Center Distance': price_vs_city_center,
    'Superhost % by City': superhost_pct_by_city,
    'Price vs Restaurant Index': price_vs_restaurant,
    'Price vs Attraction Index': price_vs_attraction,
    'Correlation Matrix': correlation_matrix,
    'Price Boxplot': price_boxplot,
}


class ThemedToolbar(NavigationToolbar2Tk):
    """matplotlib toolbar that works with a dark background.

    On Windows, Tk draws disabled icon buttons with a white checkered pattern,
    which looks broken on a dark background. In dark mode the Back/Forward
    buttons therefore stay enabled (they do nothing when there is no history).
    """
    dark = False

    def set_history_buttons(self):
        if self.dark:
            for name in ('Back', 'Forward'):
                if name in self._buttons:
                    self._buttons[name]['state'] = tk.NORMAL
        else:
            super().set_history_buttons()


class AirbnbGraphViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Airbnb Data Analysis Graphs')
        self.geometry('1250x780')
        self.minsize(900, 600)
        self.names = list(CHARTS)
        self.dark_mode = tk.BooleanVar(value=False)

        # 'clam' is used because the native Windows theme ignores custom colors
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Sidebar with chart list
        sidebar = ttk.Frame(self, padding=8)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(sidebar, text='Graphs', font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 6))

        self.listbox = tk.Listbox(sidebar, width=32, activestyle='none', font=('Segoe UI', 10),
                                  selectbackground=BAR_COLOR, exportselection=False)
        for name in self.names:
            self.listbox.insert(tk.END, name)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind('<<ListboxSelect>>', lambda e: self.show_selected())

        nav = ttk.Frame(sidebar)
        nav.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(nav, text='◀ Previous', command=lambda: self.step(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text='Next ▶', command=lambda: self.step(1)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Checkbutton(sidebar, text='Dark mode  (Ctrl+D)', variable=self.dark_mode,
                        command=self.apply_theme).pack(anchor='w', pady=(10, 0))

        # Plot area
        plot_area = ttk.Frame(self)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(9, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.toolbar = ThemedToolbar(self.canvas, plot_area)  # zoom / pan / save
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.bind('<Up>', lambda e: self.step(-1))
        self.bind('<Down>', lambda e: self.step(1))
        self.bind('<Left>', lambda e: self.step(-1))
        self.bind('<Right>', lambda e: self.step(1))
        self.bind('<Control-d>', lambda e: self.toggle_dark_mode())

        self.apply_theme(redraw=False)
        self.select(0)

    @property
    def theme(self):
        return 'dark' if self.dark_mode.get() else 'light'

    def toggle_dark_mode(self):
        self.dark_mode.set(not self.dark_mode.get())
        self.apply_theme()

    def apply_theme(self, redraw=True):
        c = THEMES[self.theme]

        self.configure(bg=c['bg'])
        self.style.configure('.', background=c['bg'], foreground=c['fg'])
        self.style.configure('TButton', background=c['button'], bordercolor=c['button'])
        self.style.map('TButton', background=[('active', c['button_active'])])
        self.style.map('TCheckbutton', background=[('active', c['bg'])],
                       indicatorbackground=[('selected', BAR_COLOR), ('!selected', c['panel'])])

        self.listbox.configure(bg=c['panel'], fg=c['fg'], selectforeground='white',
                               highlightbackground=c['bg'], highlightcolor=c['bg'])

        # Toolbar buttons are plain tk widgets, so they are colored one by one
        self.toolbar.configure(bg=c['bg'])
        for widget in self.toolbar.winfo_children():
            if isinstance(widget, (tk.Button, tk.Checkbutton, tk.Label)):
                widget.configure(bg=c['bg'], fg=c['fg'], activebackground=c['button_active'],
                                 activeforeground=c['fg'])
            if isinstance(widget, tk.Checkbutton):
                widget.configure(selectcolor=c['button_active'])
        self.toolbar.dark = self.dark_mode.get()
        self.toolbar._rescale()  # reloads the icons in light or dark versions to match
        self.toolbar.set_history_buttons()
        self.canvas.get_tk_widget().configure(bg=c['figure'])

        if redraw:
            self.show_selected()

    def select(self, index):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.show_selected()

    def step(self, delta):
        current = self.listbox.curselection()
        index = (current[0] if current else 0) + delta
        self.select(index % len(self.names))
        return 'break'

    def show_selected(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        self.figure.clear()
        with matplotlib.rc_context(chart_rc(self.theme)):
            self.figure.set_facecolor(THEMES[self.theme]['figure'])
            CHARTS[self.names[selection[0]]](self.figure)
            self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    AirbnbGraphViewer().mainloop()
