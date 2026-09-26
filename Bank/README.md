# پیش‌بینی پذیرش سپرده بانکی مدت‌دار (Bank Marketing — End-to-End ML Project)

## 1. Project Overview

این پروژه یک پروژه‌ی کامل Machine Learning است که یک مسئله‌ی واقعی بازاریابی بانکی را از ابتدا (تعریف مسئله) تا انتها (مدل نهایی و تحلیل خطا) پیش می‌برد:

```
Problem → Data → Preprocessing → Features → Model → Evaluation → Error Analysis → Conclusion
```

مسیر مسئله: **Classification** (طبقه‌بندی دودویی).

## 2. Dataset Source

- **نام دیتاست:** Bank Marketing Dataset (فایل `bank-full.csv`)
- **منبع:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- داده مربوط به کمپین‌های بازاریابی تلفنی مستقیم یک بانک پرتغالی است که هدفشان فروش سپرده‌ی بانکی مدت‌دار (Term Deposit) بوده است.

| ویژگی | مقدار |
|---|---|
| تعداد رکورد | 45,211 |
| تعداد ستون (Feature + Target) | 17 |
| تعداد Feature نهایی مدل | 15 |
| Missing Values | 0 |
| Duplicate Records | 0 |

## 3. Problem Definition

- **Problem:** پیش‌بینی اینکه آیا یک مشتری در پاسخ به تماس بازاریابی، در سپرده‌ی بانکی مدت‌دار ثبت‌نام می‌کند یا نه.
- **Target:** ستون `y` (مقادیر `yes` / `no`).
- **نوع مسئله:** **Classification** (نه Regression)، چون Target یک متغیر گسسته‌ی دودویی (دو کلاس) است، نه یک مقدار پیوسته.
- **واحد هر Sample:** یک تماس بازاریابی با یک مشتری در یک کمپین مشخص.
- **Featureهای در اختیار مدل (15 مورد):**
  - عددی: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
  - Categorical: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **آنچه پیش‌بینی می‌شود:** احتمال/برچسب `yes` یا `no` بودن پاسخ مشتری، **پیش از برقراری تماس** (بنابراین ستون `duration` که فقط بعد از پایان تماس مشخص می‌شود، از Featureها حذف شده — جزئیات در بخش Data Leakage).
- **Class Distribution (Imbalance):** `no` = 39,922 (88.3٪) در برابر `yes` = 5,289 (11.7٪) — یک مسئله‌ی نامتوازن (Imbalanced Classification).

## 4. Train / Validation / Test Split

- تقسیم سه‌بخشی با `train_test_split` و `stratify=y` (برای حفظ نسبت کلاس‌ها در هر بخش):
  - **Train:** 70٪ → 31,647 رکورد
  - **Validation:** 15٪ → 6,782 رکورد
  - **Test:** 15٪ → 6,782 رکورد
- نسبت کلاس‌ها در هر سه بخش تقریباً یکسان حفظ شده (~88.3٪ no / ~11.7٪ yes در Train، Validation و Test).
- **Random Seed:** `RANDOM_STATE = 42` در تمام مراحل (split و مدل‌ها) برای تکرارپذیری کامل نتایج.
- **قانون رعایت‌شده:** Test Set هیچ‌جا در فرآیند آموزش، Preprocessing یا انتخاب مدل استفاده نشده و فقط **یک‌بار** در انتهای پروژه برای ارزیابی نهایی به کار رفته است.

## 5. Data Preprocessing

Pipeline کاملاً reproducible با `Pipeline` و `ColumnTransformer` از scikit-learn ساخته شده:

- **Numeric Pipeline** (`age`, `balance`, `day`, `campaign`, `pdays`, `previous`):
  - `SimpleImputer(strategy="median")` برای مقادیر گم‌شده
  - `StandardScaler()` برای استانداردسازی
- **Categorical Pipeline** (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`):
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- هر دو Pipeline در یک `ColumnTransformer` ترکیب شده‌اند.
- **نکته‌ی کلیدی:** `fit` روی Preprocessing فقط روی `X_train` انجام می‌شود؛ `X_val` و `X_test` فقط `transform` می‌شوند (نه `fit_transform`) تا هیچ اطلاعاتی از Validation/Test به پارامترهای Preprocessing (مثل میانگین/انحراف‌معیار Scaler یا دسته‌های Encoder) نشت نکند.
- بعد از Preprocessing، هیچ مقدار گم‌شده‌ای در Train، Validation یا Test باقی نمانده است (بررسی و تأیید شده).
- خروجی نهایی Preprocessing برای هر نمونه، بردار 50 بعدی است (6 ستون عددی + حاصل One-Hot Encoding ستون‌های categorical).

## 6. Data Leakage

**Data Leakage چیست؟** وقتی اطلاعاتی از خارج از داده‌ی واقعاً در دسترسِ زمان پیش‌بینی (یا از Validation/Test) وارد فرآیند آموزش مدل شود، به‌طوری‌که مدل در عمل به چیزی دسترسی داشته باشد که در دنیای واقعی هنگام Prediction وجود ندارد.

**چرا خطرناک است؟** چون مدل عملکردی کاذب و بیش‌ازحد خوش‌بینانه روی داده‌ی آموزش/آزمایش نشان می‌دهد، اما در استقرار واقعی (Production) که آن اطلاعات در دسترس نیست، عملکردش به‌شدت افت می‌کند.

**مثال واقعی از این پروژه:** ستون `duration` (مدت زمان تماس تلفنی بر حسب ثانیه) در دیتاست خام وجود دارد و همبستگی بسیار قوی با `y` دارد (تماس‌های طولانی‌تر معمولاً به «yes» ختم می‌شوند). اما این مقدار **فقط بعد از پایان تماس** مشخص می‌شود؛ در سناریوی واقعی که مدل باید *قبل از* تماس گرفتن با مشتری، احتمال موفقیت را پیش‌بینی کند، این عدد اصلاً در دسترس نیست. اگر `duration` در مدل باقی می‌ماند، Accuracy/F1 گزارش‌شده به‌طور کاذب بسیار بالا می‌شد ولی مدل در استفاده‌ی واقعی (پیش از تماس) قابل استفاده نبود.

**چگونه جلوی آن گرفته شد؟** ستون `duration` در همان ابتدای کار — قبل از Train/Test Split، Preprocessing و آموزش هر مدلی — به‌طور کامل از Feature Set حذف شد (`X = X.drop(columns=["duration"])`)، در تمام نوت‌بوک‌ها (Problem_Data، preprocessing، baseline، model_training، evaluation) به‌طور یکسان.

**نقطه‌ی بالقوه‌ی دیگر Leakage که رعایت شده:** `fit` مربوط به Imputer/Scaler/Encoder فقط روی Train انجام شده (بخش 5)، و Test Set تا انتهای پروژه برای هیچ تصمیمی (انتخاب مدل، Hyperparameter Tuning) استفاده نشده است.

## 7. Baseline Model

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy Classifier (most_frequent) | 0.8829 | 0.0000 | 0.0000 | 0.0000 | — |
| Logistic Regression | 0.8931 | 0.6575 | 0.1814 | 0.2843 | 0.7699 |

- **Baseline چیست؟** `DummyClassifier(strategy="most_frequent")` همیشه پرتکرارترین کلاس (`no`) را پیش‌بینی می‌کند و هیچ الگویی از داده یاد نمی‌گیرد. علاوه بر آن، `Logistic Regression` هم به‌عنوان یک baseline دوم و آموزنده‌تر (یک مدل خطی ساده) اجرا شد.
- **چرا این دو؟** Dummy Classifier کف مطلق عملکرد را نشان می‌دهد (چیزی که هر مدل واقعی باید از آن بهتر باشد) و Logistic Regression سطح یک مدل ساده‌ی خطی را نشان می‌دهد.
- **نکته‌ی مهم:** Dummy Classifier با Accuracy=88.3٪ (فقط با حدس زدن «no» برای همه) نشان می‌دهد که **Accuracy به‌تنهایی در این مسئله گمراه‌کننده است** — یک مدل کاملاً بی‌فایده هم Accuracy بالایی می‌گیرد، چون کلاس‌ها نامتوازن‌اند. مدل اصلی باید علاوه بر Accuracy، در Precision/Recall/F1/ROC-AUC هم به‌وضوح بهتر از این دو Baseline باشد.

## 8. Machine Learning Models

سه الگوریتم زیر پیاده‌سازی شدند:

1. **Logistic Regression** — یک مدل خطی ساده و تفسیرپذیر؛ به‌عنوان مبنای مقایسه با مدل‌های پیچیده‌تر.
2. **Decision Tree** (`max_depth=8`) — یک مدل غیرخطی که می‌تواند تعامل بین Featureها را یاد بگیرد، بدون نیاز به Scaling.
3. **Random Forest** (`n_estimators=200`) — یک مدل Ensemble از تعداد زیادی درخت تصمیم که معمولاً پایدارتر و دقیق‌تر از یک درخت تنها عمل می‌کند.

**چرا این سه مدل؟** دیتاست ترکیبی از Featureهای عددی و categorical دارد، رابطه‌های احتمالاً غیرخطی بین برخی Featureها (مثل `poutcome`, `month`) و Target وجود دارد، و کلاس‌ها نامتوازن‌اند. این سه مدل طیفی از سادگی به پیچیدگی را پوشش می‌دهند (خطی ساده → درخت تکی → Ensemble)، که امکان مقایسه‌ی معنادار بین سطوح مختلف پیچیدگی مدل را فراهم می‌کند.

## 9. Evaluation

Metricهای استفاده‌شده و دلیل انتخابشان:

| Metric | چه چیزی اندازه می‌گیرد؟ | چرا برای این مسئله مهم است؟ |
|---|---|---|
| Accuracy | درصد کل پیش‌بینی‌های درست | فقط برای تصویر کلی؛ به‌تنهایی کافی نیست |
| Precision | از بین کسانی که مدل «yes» پیش‌بینی کرده، چند درصد واقعاً «yes» بودند | نشان می‌دهد تماس‌های تبلیغاتی هدررفته (روی مشتریانی که واقعاً «no» بودند) چقدرند |
| Recall | از بین همه‌ی مشتریان واقعاً «yes»، چند درصد شناسایی شدند | نشان می‌دهد چه تعداد مشتری بالقوه‌ی علاقه‌مند از دست رفته‌اند (فرصت‌های ازدست‌رفته) |
| F1 | میانگین هارمونیک Precision و Recall | معیار تعادلی؛ چون هم Precision پایین (اتلاف منابع) و هم Recall پایین (از‌دست‌دادن فرصت) هزینه دارند |
| ROC-AUC | توانایی مدل در رتبه‌بندی صحیح مشتریان بر اساس احتمال «yes»، مستقل از آستانه‌ی تصمیم | برای این‌که بانک بتواند مشتریان را بر اساس اولویت (نه فقط برچسب دودویی) تماس بگیرد |
| Confusion Matrix | تعداد دقیق TP/TN/FP/FN | پایه‌ی تحلیل خطا (بخش 12) |

**چه زمانی Accuracy گمراه‌کننده است؟** دقیقاً همین‌جا: چون فقط 11.7٪ از مشتریان واقعاً «yes» هستند، مدلی که همیشه «no» پیش‌بینی کند بدون یادگیری هیچ الگویی، 88.3٪ Accuracy می‌گیرد (به‌وضوح در نتیجه‌ی Dummy Classifier دیده شد) — بنابراین Accuracy بالا لزوماً به‌معنای مدل خوب نیست و باید همراه با Recall/F1/ROC-AUC خوانده شود.

## 10. Model Comparison (روی Validation)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8931 | 0.6575 | 0.1814 | 0.2843 | 0.7699 |
| Decision Tree (depth=8) | 0.8960 | 0.6787 | 0.2128 | 0.3241 | 0.7151 |
| Random Forest (n=200) | 0.8963 | 0.6454 | 0.2544 | 0.3650 | 0.7930 |

**تفسیر رفتار هر مدل:**
- هر سه مدل از نظر Accuracy تقریباً یکسان و فقط کمی بالاتر از Dummy Baseline (88.3٪) هستند — دوباره نشان می‌دهد Accuracy معیار خوبی برای تفکیک این مدل‌ها نیست.
- **Random Forest** بالاترین Recall (0.254) و بالاترین ROC-AUC (0.793) را دارد — بهترین توانایی در شناسایی مشتریان واقعاً علاقه‌مند و رتبه‌بندی صحیح آن‌ها.
- **Decision Tree** بالاترین Precision (0.679) را دارد اما پایین‌ترین ROC-AUC (0.715) — یعنی وقتی «yes» پیش‌بینی می‌کند نسبتاً مطمئن‌تر است، اما در رتبه‌بندی کلی مشتریان ضعیف‌تر عمل می‌کند.
- **Logistic Regression** پایین‌ترین Recall (0.181) را دارد — بیشترین تعداد مشتری واقعاً علاقه‌مند را از دست می‌دهد، احتمالاً چون نمی‌تواند روابط غیرخطی موجود در Featureهایی مثل `poutcome` را به‌خوبی مدل‌های درختی یاد بگیرد.
- انتخاب مدل صرفاً بر اساس یک عدد انجام نشده؛ ترکیب F1 (تعادل) و ROC-AUC (رتبه‌بندی) ملاک اصلی مقایسه بوده‌اند.

## 11. Hyperparameter Tuning

- **مدل هدف:** Random Forest.
- **روش:** `GridSearchCV` با `cv=3` و معیار امتیازدهی سفارشی `F1` (با `pos_label="yes"`، چون کلاس مثبت اقلیت است).
- **Search Space:**

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, 12],
    "model__min_samples_leaf": [1, 2]
}
```

- **چرا این Hyperparameterها؟** `n_estimators` تعداد درخت‌ها (تعادل دقت/سرعت)، `max_depth` عمق درخت‌ها (کنترل Overfitting)، `min_samples_leaf` حداقل نمونه در هر برگ (کنترل بیشتر Overfitting) — سه پارامتر اصلی و پراستفاده برای تنظیم Random Forest.
- **بهترین ترکیب یافته‌شده:** `max_depth=12`, `min_samples_leaf=1`, `n_estimators=100` (بهترین F1 روی Cross-Validation: 0.2737).

**قبل و بعد از Tuning (روی Validation):**

| Metric | قبل از Tuning | بعد از Tuning |
|---|---|---|
| Accuracy | 0.8963 | 0.8963 |
| Precision | 0.6454 | 0.7136 |
| Recall | 0.2544 | 0.1914 |
| F1 | 0.3650 | 0.3019 |
| ROC-AUC | 0.7930 | 0.7999 |

**تفسیر:** برخلاف انتظار رایج، Tuning باعث بهبود F1 نشد — Precision و ROC-AUC کمی بهتر شدند، اما Recall و F1 به‌طور محسوس افت کردند. این نشان می‌دهد مدل تنظیم‌شده محافظه‌کارتر عمل کرده (کمتر «yes» پیش‌بینی می‌کند، با اطمینان بیشتر)، در حالی که هدف اصلی (F1 روی کلاس اقلیت) عملاً کاهش یافت. به همین دلیل، نسخه‌ی Tuning‌شده به‌عنوان مدل نهایی انتخاب **نشد** (جزئیات در بخش 14).

## 12. Error Analysis

بررسی بر اساس Confusion Matrix روی Validation:

| | Predicted: no | Predicted: yes |
|---|---|---|
| **Actual: no** | 5,927 (TN) | 61 (FP) |
| **Actual: yes** | 642 (FN) | 152 (TP) |

- **معنای False Negative (642 مورد):** مشتریانی که واقعاً می‌خواستند سپرده باز کنند ولی مدل پیش‌بینی کرد «no» — یعنی **فرصت فروش ازدست‌رفته**. این پرهزینه‌ترین نوع خطا برای بانک است.
- **معنای False Positive (61 مورد):** مشتریانی که مدل «yes» پیش‌بینی کرد ولی واقعاً «no» بودند — یعنی **هزینه‌ی تماس/بازاریابی هدررفته** روی مشتری‌ای که تبدیل نشده.
- **Class Imbalance به‌وضوح در خطاها منعکس است:** تعداد False Negative (642) بسیار بیشتر از False Positive (61) است؛ مدل به‌طور سیستماتیک به سمت پیش‌بینی کلاس اکثریت («no») متمایل است — نتیجه‌ی مستقیم نامتوازن بودن داده (88.3٪ در برابر 11.7٪).

**حداقل 3 مورد بررسی گروهی خطا:**

1. **بر اساس شغل (`job`):** بالاترین نرخ خطا در `student` (18.9٪) و `retired` (22.8٪→18.7٪ نرخ خطا) دیده می‌شود، در حالی که همین دو گروه طبق EDA اولیه بالاترین نرخ پذیرش واقعی (`student`: 28.7٪, `retired`: 22.8٪) را داشتند. یعنی مدل دقیقاً روی گروه‌هایی که بیشترین شانس واقعی «yes» گفتن را دارند، بیشترین اشتباه را می‌کند — چون این گروه‌ها در داده‌ی آموزش تعداد نمونه‌ی کمتری دارند (`student`: 938 نفر، `retired`: 2,264 نفر از کل 45,211) و مدل الگوی رفتاری آن‌ها را ضعیف‌تر یاد گرفته.
2. **بر اساس تحصیلات (`education`):** گروه `unknown` (13.6٪) و `tertiary` (12.2٪) بیشترین نرخ خطا، و `primary` (7.9٪) کمترین نرخ خطا را دارد.
3. **بر اساس وام مسکن (`housing`):** مشتریانی که وام مسکن ندارند (`housing=no`) نرخ خطای 14.0٪ دارند، تقریباً دو برابر مشتریانی که وام مسکن دارند (`housing=yes`، نرخ خطا 7.4٪) — نشان می‌دهد الگوی رفتاری این دو گروه به‌وضوح متفاوت است و مدل در گروه بدون وام مسکن نااطمینان‌تر عمل می‌کند.

**نمونه‌های واقعی False Negative:** سه مشتری با احتمال پیش‌بینی‌شده‌ی «yes» بسیار پایین (بین 0.035 تا 0.443) در حالی که واقعاً «yes» بودند — نشان می‌دهد مدل به‌شدت محافظه‌کارانه عمل می‌کند و برای اطمینان از پیش‌بینی «yes»، به آستانه‌ی احتمال بسیار بالایی نیاز دارد که خود پیامد مستقیم Class Imbalance است.

## 13. Feature Importance (Permutation Importance)

روی مدل نهایی (Random Forest)، با `permutation_importance` (معیار: F1 روی کلاس «yes»):

| رتبه | Feature | اهمیت (کاهش میانگین F1) |
|---|---|---|
| 1 | poutcome | 0.1857 |
| 2 | month | 0.0888 |
| 3 | pdays | 0.0686 |
| 4 | contact | 0.0458 |
| 5 | housing | 0.0372 |
| 6 | previous | 0.0320 |
| 7 | job | 0.0312 |
| 8 | campaign | 0.0200 |
| 9 | day | 0.0197 |
| 10 | education | 0.0157 |

**تفسیر:** `poutcome` (نتیجه‌ی کمپین بازاریابی قبلی) با فاصله‌ی زیاد مهم‌ترین Feature است — مشتری‌ای که در کمپین قبلی «yes» گفته، به‌طور قابل‌توجهی محتمل‌تر است دوباره «yes» بگوید. `month` و `pdays` (فاصله از آخرین تماس) هم نقش قابل‌توجهی دارند که می‌تواند بازتاب فصلی بودن اثربخشی کمپین‌ها یا تازگی ارتباط با مشتری باشد.

**⚠️ Feature Importance ≠ Causality:** بالا بودن اهمیت `poutcome` به این معنا نیست که «نتیجه‌ی کمپین قبلی» علت رفتار مشتری در کمپین فعلی است؛ صرفاً نشان می‌دهد مشتریانی با سابقه‌ی تعامل مثبت، یک بخش (Segment) طبیعتاً پذیراتر از جمعیت هستند. این یک همبستگی پیش‌بینی‌کننده است، نه یک رابطه‌ی علّی اثبات‌شده.

## 14. Final Model

- **مدل نهایی انتخاب‌شده:** **Random Forest** با `n_estimators=200` (نسخه‌ی پیش از Tuning، نه نسخه‌ی GridSearchCV).
- **معیار انتخاب:** F1-Score روی Validation Set — چون در این مسئله‌ی نامتوازن، تعادل بین Precision و Recall (نه فقط Accuracy) اهمیت دارد.
- **چرا این مدل؟** در مقایسه‌ی بخش 10، Random Forest (پیش از Tuning) بالاترین F1 (0.365) را در میان همه‌ی مدل‌های بررسی‌شده داشت — حتی بالاتر از نسخه‌ی Tuning‌شده‌ی خودش (F1=0.302، بخش 11). چون هدف پروژه صرفاً بالا بردن یک عدد (مثلاً Precision یا Accuracy) نبود، بلکه معیار تعادلی F1 بود، نسخه‌ی Tuning‌نشده که این معیار را بهتر برآورده می‌کرد، به‌عنوان مدل نهایی نگه داشته شد.
- **نتیجه‌ی نهایی روی Test Set** (که تا این مرحله برای هیچ تصمیمی استفاده نشده بود):

| Metric | Test |
|---|---|
| Accuracy | 0.8977 |
| Precision | 0.6656 |
| Recall | 0.2509 |
| F1 | 0.3645 |
| ROC-AUC | 0.7801 |

Confusion Matrix روی Test: TN=5,889، FP=100، FN=594، TP=199.

نزدیکی این اعداد به نتایج Validation (Accuracy 0.896، F1 0.365، ROC-AUC 0.793) نشان می‌دهد مدل روی داده‌ی دیده‌نشده هم رفتار پایداری دارد و بیش‌برازش قابل‌توجهی روی Validation صورت نگرفته است.

**محدودیت‌های مدل:**
- Recall پایین (~25٪) یعنی حدود سه‌چهارم مشتریانی که واقعاً «yes» می‌گفتند، همچنان شناسایی نمی‌شوند.
- حذف عمدی `duration` (برای جلوگیری از Leakage) یک سقف عملکردی واقع‌گرایانه اما پایین‌تر برای مدل ایجاد می‌کند؛ این محدودیت آگاهانه پذیرفته شده چون بازتاب شرایط واقعی پیش‌بینی است.
- Search Space مربوط به Hyperparameter Tuning محدود بود (فقط 8 ترکیب)؛ گسترش آن یا استفاده از راهکارهای مواجهه با Imbalance (مثل `class_weight="balanced"` یا SMOTE) می‌تواند Recall/F1 را بهبود دهد.
- داده مربوط به کمپین‌های یک بانک پرتغالی در یک بازه‌ی زمانی خاص است؛ تعمیم مستقیم به بازارها یا زمان‌های دیگر بدون آموزش مجدد توصیه نمی‌شود.

**آیا این مدل آماده‌ی استفاده‌ی واقعی است؟** به‌طور کامل نه. مدل برای **اولویت‌بندی مشتریان** (رتبه‌بندی بر اساس احتمال پیش‌بینی‌شده و تماس با نفرات بالای لیست) قابل استفاده است، اما به‌دلیل Recall پایین نباید به‌عنوان فیلتر قطعی «تماس بگیر / نگیر» به‌کار رود؛ پیش از استقرار کامل عملیاتی، بهبودهایی مثل مدیریت بهتر Class Imbalance و گسترش Hyperparameter Search توصیه می‌شود.

## 15. Reproducibility

- **requirements.txt:** فایل جداگانه در ریشه‌ی پروژه.
- **Random Seed:** `RANDOM_STATE = 42` به‌صورت ثابت در تمام نوت‌بوک‌ها (Split و همه‌ی مدل‌ها).
- **Pipeline:** تمام مراحل Preprocessing + Model در یک `sklearn.pipeline.Pipeline` واحد قرار دارند؛ آموزش، Predict و Tuning همگی روی همین Pipeline انجام می‌شود، نه روی داده‌ی از‌پیش‌تبدیل‌شده.
- **ساختار پروژه (فعلی):**

```
Bank/
├── bank-full.csv          # دیتاست خام
├── Problem_Data.ipynb     # تعریف مسئله + Data Understanding
├── preprocessing.ipynb    # Split + Preprocessing Pipeline
├── baseline.ipynb         # Dummy Classifier + Logistic Regression
├── model_training.ipynb   # آموزش 3 مدل (Logistic, Decision Tree, Random Forest)
├── evaluation.ipynb       # مقایسه مدل‌ها + Tuning + Error Analysis + Feature Importance + Final Model
├── requirements.txt
└── README.md
```

- تمام نوت‌بوک‌ها فایل `bank-full.csv` را با نام مستقیم می‌خوانند؛ برای اجرا باید در همان مسیر نوت‌بوک‌ها قرار داشته باشد.

## 16. How to Run

```bash
# 1) ساخت محیط مجازی (اختیاری ولی توصیه‌شده)
python -m venv venv
source venv/bin/activate      # ویندوز: venv\Scripts\activate

# 2) نصب وابستگی‌ها
pip install -r requirements.txt

# 3) اجرای نوت‌بوک‌ها به همین ترتیب
jupyter notebook Problem_Data.ipynb    # تعریف مسئله و شناخت داده
jupyter notebook preprocessing.ipynb   # Split و ساخت Pipeline
jupyter notebook baseline.ipynb        # Baseline Models
jupyter notebook model_training.ipynb  # آموزش 3 مدل اصلی
jupyter notebook evaluation.ipynb      # مقایسه، Tuning، Error Analysis، Feature Importance، Final Model
```

> هر نوت‌بوک با **Kernel → Restart & Run All** بدون خطا از ابتدا تا انتها اجرا می‌شود.

## References

- Dataset: [Bank Marketing – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- کتابخانه‌ها: [pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/documentation.html) (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `permutation_importance`)، [matplotlib](https://matplotlib.org/stable/)
