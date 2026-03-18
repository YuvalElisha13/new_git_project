import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# קריאה לנתונים (דורש את הקובץ titanic.csv)
df = pd.read_csv("titanic.csv")

# הצגת חמש השורות הראשונות
print("------ חמש השורות הראשונות ------")
print(df.head())

# סקירת מידע כללי
print("\n------ מידע כללי ------")
print(df.info())

# ניתוח נתונים חסרים
print("\n------ נתונים חסרים ------")
print(df.isnull().sum())

# סטטיסטיקות תיאוריות
print("\n------ סטטיסטיקות תיאוריות ------")
print(df.describe(include='all'))

# התפלגות הנוסעים ששרדו
survived_counts = df['Survived'].value_counts()
print("\n------ התפלגות שורדים ------")
print(survived_counts)

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('התפלגות שורדים')
plt.xticks([0,1], ['לא שרד', 'שרד'])
plt.show()

# התפלגות המגדרים
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('התפלגות שורדים לפי מגדר')
plt.show()

# התפלגות גילאים
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('התפלגות גילאים')
plt.xlabel('גיל')
plt.show()

# גיל ממוצע של שורדים/לא שורדים
print("\n------ גיל ממוצע של שורדים מול לא שורדים ------")
print(df.groupby("Survived")["Age"].mean())

# פילוח לפי מחלקה
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('התפלגות שורדים לפי מחלקה')
plt.show()

# שרידות לפי מחיר כרטיס (Fare)
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.xticks([0,1], ['לא שרד', 'שרד'])
plt.title('מחיר כרטיס מול שרידות')
plt.show()

# קורלציות בין מאפיינים נומריים
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('מטריצת קורלציה')
plt.show()

# מסקנות קצרות
print("\n------ מסקנות ביניים ------")
print("הניתוח מראה:\n"
      "- שיעור השרידות של נשים גבוה משמעותית.\n"
      "- המחלקה הראשונה והשנייה עם שיעורי שרידות גבוהים יותר.\n"
      "- גיל ומחיר כרטיס משחקים תפקיד מסוים בשרידות.\n"
      "- קיימים נתונים חסרים בעיקר בעמודת הגיל.")



