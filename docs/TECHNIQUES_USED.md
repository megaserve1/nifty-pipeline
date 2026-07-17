# Every technique this pipeline uses — plain-words reference

_For answering the manager. Each item: the real name, what it does in plain words, why we chose it, the likely question, and where it lives in the code. Generated from the actual code, not theory._

## Quick index

- **The three models** — Random Forest classifier · XGBoost gradient-boosted trees, objective multi:softprob · CatBoost classifier, loss_function MultiClass, oblivious · Multiclass prediction via per-class probabilities + argmax · Per-row instance weighting via sample_weight · Model-specific missing-value handling: native NaN
- **Handling class imbalance (weighting)** — Per-row sample_weight · Signal-strength / conviction weighting · Cost-sensitive class weighting fixed in the label policy · No resampling · Weight-aware leaf floor · Macro-averaged, threshold-free evaluation
- **How we score the models** — trading_cost · Severity matrix · Macro-averaged F1 · PR-AUC per class · Mean PR-AUC · Per-class precision and recall · Accuracy deliberately banned · Blind-class disqualification
- **Categorical encoding** — Ordinal · Unknown / unseen category handling · Fit-on-train-only preprocessing · Encoder persistence / train-serve parity · CatBoost native categorical handling · Missing category as an explicit 'MISSING' label · Target label encoding via sklearn LabelEncoder
- **Missing values** — Sentinel-value imputation · Per-column computed sentinel · Flat fixed sentinel -999 · Native missing-value handling · Explicit missing category · Bounded forward-fill · Drop-as-mask · Fit-on-train-only imputation
- **No-lookahead time alignment** — Point-in-time as-of join on bar close · Session-anchored bar boundaries · Empirical clock detection -- measure each column's real bar period instead of trusting the declaration · Slowest-fit clock rule -- the LARGEST period the column never moves inside · Per-column clock alignment -- group columns by clock, align each group on its own bar-close · Staleness tolerance / forward-fill cap · Fail-closed on timezone-aware timestamps · Relative-tolerance constancy test, fail-slow on NaN and bool
- **The leak guard** — Lookahead name denylist · Target-leakage name ban, hard-block plus soft-warn · Behavioural leak test: forward-return vs past-return correlation · Calendar / identifier memorisation guard · Report-only · Unconditional calendar drop · Auditable allowlist override · Defence in depth: screen the parquet, not just the registry · Deterministic categorical encoding
- **Train / validation / test split & cross-validation** — Time-based · Separate validation set for tuning; test opened once · Embargo / purge gap between slices, counted in TRADING SESSIONS · Purged & embargoed k-fold cross-validation, asymmetric cuts · Post-split leakage assertion
- **Hyperparameter optimisation (HPO)** — Random Search · Grid Search · ClearML HyperParameterOptimizer · Custom objective · Hold-out validation for tuning · Parameter search space · Log-uniform · Result caching keyed by dataset content hash · Pinning constants through HPO · Trial budget and per-job time cap · Parallel trials capped to the number of agents · HPO preflight · Manual winner promotion
- **Feature selection** — Manual expert ballot · Group / block selection · Explicit named-set selection · Leave-one-out feature ablation
- **Data & experiment versioning** — Recipe-not-copy · Semantic major.minor versioning for single-variable ablation · Content-addressed hashing · DVC content-addressed data layer with ClearML external-file pointer · Manifest certificate with a validate-on-publish gate · No-lookahead sworn statement · ClearML dataset versioning with mandatory finalize()+publish() ordering and reuse-repair · Lock-back receipt · Immutable-version byte guard
- **Explainability (SHAP)** — SHAP · Cost-weighted error ranking · Tail-risk / worst-case screen · Mean-absolute-SHAP feature importance, normalized to shares · Bootstrap stability of feature shares · SHAP waterfall plot · Stratified sampling of the worst pair


---

## 1. The three models

### Random Forest classifier (bagging / bootstrap-aggregated decision trees)
Builds 250 separate decision trees. Each tree sees a random slice of the rows and only about 22% of the columns at each split, so the trees end up different from each other. Every tree is grown deep (depth 22). To predict, all trees vote and the votes are averaged. Averaging many deep, disagreeing trees cancels out the noise.

**Why this, not the alternative:** The forest is grown DEEP and averaged, not shallow. Depth used to arrive as 6 (a boosting depth) shared across all three models, which quietly crippled the forest. A forest wants strong deep trees that get averaged; boosting wants shallow trees that correct each other. The real brake here is min_samples_leaf=30, not max_depth, and it also forces the trees to disagree so averaging actually helps.

**Q (manager):** Why is your forest depth 22 when your xgboost is depth 4? Isn't deep overfitting?  
**A:** A forest averages independent deep trees to cancel noise, so it wants each tree strong and deep. Boosting stacks shallow trees that fix each other, so it wants them shallow. Putting the same depth on both crippled the forest -- that was a real one-day bug. The forest's actual regulariser is min_samples_leaf=30, not depth.

`trainer/train.py:110-128; defaults configs/hyperparams.yaml:28-40`

### XGBoost gradient-boosted trees, objective multi:softprob (7-class)
Builds 1000 shallow trees one after another. Each new tree fixes the mistakes of the trees before it. The learning rate 0.03 keeps every step small so it learns slowly and steadily. The objective multi:softprob makes it output one probability for each of the 7 classes.

**Why this, not the alternative:** The reference PDF said binary:logistic, but that is a 2-class objective. We have 7 classes, so it must be multi:softprob with num_class=7. Depth is kept shallow at 4 on purpose: deep trees would memorise 1-minute market noise. tree_method is 'hist', which is fast and handles missing values itself.

**Q (manager):** The proposed-hyperparameter PDF says objective binary:logistic -- why did you override it?  
**A:** binary:logistic is for two classes only. We predict one of seven classes per minute, so it has to be multi:softprob with num_class=7, which gives one probability per class. Using the 2-class objective would be flatly wrong, not a tuning choice.

`trainer/train.py:130-155 (objective at line 146, num_class at 149); defaults configs/hyperparams.yaml:48-65`

### CatBoost classifier, loss_function MultiClass, oblivious (symmetric) trees with native categoricals
Another boosting model, 1200 trees. Two things make it different from xgboost: it reads text columns (like gap_state = NO_GAP / GAP_UP / GAP_DOWN) directly with no conversion, and each tree uses the SAME split on every node of a level (an oblivious tree). loss_function MultiClass is its 7-class setting.

**Why this, not the alternative:** It is in the lineup precisely because it handles text columns natively and grows symmetric trees -- a genuinely different model from xgboost, so training all three and picking a champion hedges the modelling choice. Its regularisation was raised off the loose PDF values (l2_leaf_reg 0.5 to 3.0, random_strength 0.2 to 1.0, bagging_temperature 0 to 1.0) because loose brakes memorise noisy 1-minute data. Depth is capped at 16 because the library refuses more.

**Q (manager):** You already have xgboost -- both are boosting. Why also catboost?  
**A:** They differ on two things that matter here: catboost reads categorical text columns natively and grows oblivious (symmetric) trees, while xgboost needs text encoded to numbers and grows asymmetric trees. Different models make different mistakes, so we train all three and let select_champion pick the best on the held-out test.

`trainer/train.py:157-172 (loss_function at line 167); defaults configs/hyperparams.yaml:77-96`

### Multiclass prediction via per-class probabilities + argmax (with a full-probability guard)
The 7 text labels are turned into numbers 0-6 by a LabelEncoder. Each model outputs a probability for each of the 7 classes. The predicted class is simply the one with the highest probability (argmax). full_proba() makes sure there are always 7 probability columns, even when a rare class never appeared in the training slice.

**Why this, not the alternative:** We keep the full probability vector (softprob) rather than just the hard label, because ranking the rare entry classes with PR-AUC needs the score, not just the winning class. full_proba fixes a silent sklearn RandomForest bug: if a class is missing from training, the forest returns fewer columns and every later class's probability silently shifts to the wrong class.

**Q (manager):** How does a tree model give 7 classes at once, and what happens if a class never shows up in a training slice?  
**A:** Each model emits one probability per class and we take the argmax. sklearn's forest only returns columns for classes it actually saw, so full_proba re-inserts any missing class as a zero column. Without that, a short run would misalign every probability and no error would show.

`trainer/train.py:379-381 (LabelEncoder), 178-198 (full_proba), 459-473 (argmax on val and test)`

### Per-row instance weighting via sample_weight (not resampling)
Every row carries a weight that encodes signal strength. We pass those weights straight into fit(sample_weight=...), so a stronger-conviction row counts more in the loss. The exact same call works in all three libraries.

**Why this, not the alternative:** Chosen over SMOTE / over- / under-sampling because resampling breaks time order and can put the same row in both train and test. A weight of 2 is like duplicating a row without copying data and without leaking across the time split. The trainer also shouts about the trap that NO_TRADE has weight 0, so 53% of rows contribute nothing and the model over-trades -- a label-policy fix, not a trainer fix.

**Q (manager):** How do you handle the class imbalance -- 53% of rows are NO_TRADE?  
**A:** We weight each row by its signal strength through sample_weight in fit(), not resampling, because resampling breaks temporal order and leaks rows across the split. But be clear: the weight file ranks conviction, it does NOT fix imbalance. NO_TRADE sits at weight 0, so the model never learns to stay out -- that has to be fixed upstream in the label policy, around 0.1 to 0.2.

`trainer/train.py:450 (fit with sample_weight); weight column read at 344, weight-0 warning 363-376`

### Model-specific missing-value handling: native NaN (XGBoost/CatBoost) vs sentinel imputation (Random Forest)
xgboost and catboost handle missing values themselves -- at each split they try sending the missing rows left, then right, and keep whichever predicts better, so 'value is missing' becomes its own learned branch. Random forest crashes on NaN, so for it we fill each missing cell with a sentinel number set below every real value, and the tree separates the missing rows with one cut.

**Why this, not the alternative:** The RF sentinel is computed from the TRAIN rows only (so no test-period fact leaks into preparation) and set below every real value on purpose -- never 0 and never the mean, because those are values a feature can genuinely take and would collide with real rows. By hand this reproduces exactly what xgboost and catboost do natively.

**Q (manager):** Random forest can't take NaN -- did you just fill missing values with zero or the column mean?  
**A:** No. Zero and the mean are values a feature can really take, so they'd collide with real rows and hide the fact a value was missing. We fill with a sentinel below every real value, computed from train rows only, so the tree isolates the missing rows in a single cut -- the same thing xgboost and catboost do on their own.

`trainer/train.py:386-424 (catboost native at 386-392, RF sentinel at 410-422, xgboost native NaN note at 424)`


---

## 2. Handling class imbalance (weighting)

### Per-row sample_weight (instance weighting / observation weighting)
Every training row carries a number in the labels file's `weight` column. When we fit, we hand that column straight to the model as `sample_weight`. A row with weight 2 pulls on the model twice as hard as a row with weight 1, exactly as if we had copied it twice -- but we never copy anything. All three models take it the same way: `model.fit(Xtr, ytr, sample_weight=wtr)`. The forest is told `class_weight=None` on purpose so the per-row weight is the only weighting in play.

**Why this, not the alternative:** The obvious alternative is resampling the rows to balance the classes. Weighting gets the same effect as duplicating rows without actually duplicating them -- no copies, no extra memory, and it never lets the same minute land in both train and test. It is one number per row, so it can carry finer information than a class-level knob could.

**Q (manager):** How does the imbalance handling reach the model -- is it a config flag or the data?  
**A:** It rides in the data. The labels CSV has a `weight` column; the trainer loads it, slices it to the train rows, and passes it as `sample_weight` to fit. sklearn, XGBoost and CatBoost all accept that argument, so the exact same weight vector drives all three models.

`trainer/train.py:344 (load weight col), :430 (wtr = w[tr]), :450 (fit sample_weight=wtr), :127 (RF class_weight=None -> we use per-row instead)`

### Signal-strength / conviction weighting (NOT inverse-frequency weighting)
The weights rank how strong a signal is, not how rare a class is. SUPER = 0.91 (biggest conviction), SUB = 0.46 (medium), SMALL = 0.18 (smallest). So a big high-conviction trade counts for a lot and a weak trade counts for little. This is the OPPOSITE of the usual trick where you weight rare classes UP. Here the rarest class (ENTRY_SUB, 1.2%) gets a middling 0.46, not the biggest weight. The weight file ranks conviction; it does not fix class imbalance.

**Why this, not the alternative:** Standard inverse-frequency weighting would tell the model 'the rare classes matter most'. That is not what the desk wants. The desk wants the model to try hardest on the trades that move the most money -- the high-size SUPER signals -- and to shrug at weak ones. So the weight encodes business conviction, and imbalance is dealt with separately (see the NO_TRADE fix).

**Q (manager):** Do the weights just up-weight the rare classes like normal imbalance handling?  
**A:** No -- it is the reverse. Weight = signal strength: SUPER 0.91 > SUB 0.46 > SMALL 0.18. The rarest class gets a middling weight, not the biggest. The weight ranks conviction, not rarity; class imbalance is handled by the label-policy weight on NO_TRADE, not by this ladder.

`configs/severity_7class.json (_the_size_ladder: SUPER 0.91 > SUB 0.46 > SMALL 0.18); config.py:257-259 (WEIGHT_COL); CLAUDE.md label table`

### Cost-sensitive class weighting fixed in the label policy (the NO_TRADE zero-weight fix)
NO_TRADE is 53% of all rows and it used to have weight 0. A row with weight 0 adds nothing to the loss, so the model never learned to stay out and wanted to trade every single minute. The fix does NOT live in the trainer. It lives upstream in the labels file: NO_TRADE now carries 0.064 instead of 0, which is about 12.4% of the loss. The trainer's only job here is a guardrail -- it measures the share of zero-weight rows, and if any class has mean weight 0 it prints a loud warning and tags the run UNLEARNABLE_CLASS.

**Why this, not the alternative:** You could hack a weight into the trainer, but then every downstream check (the manifest, the metrics, live inference) would see one thing and training would do another. Fixing it in the label policy keeps a single source of truth: the weight that trains the model is the same weight the certificate swears to and the same weight live inference sees. The trainer stays a dumb, honest consumer that only warns.

**Q (manager):** If NO_TRADE had weight 0 and it's half the data, why didn't the model just ignore 'don't trade' -- and where did you fix it?  
**A:** It did ignore it -- with weight 0 those 53% of rows contributed nothing, so it over-traded. We fixed it upstream in the label policy: NO_TRADE is now 0.064 (~12.4% of the loss), not 0. The trainer only checks and shouts if it ever sees a zero-weight class again; it does not patch the number itself.

`trainer/train.py:363-376 (the [4/6] weight warning + UNLEARNABLE_CLASS tag); config.py:189-198 (NO_TRADE 0.000 -> 0.064)`

### No resampling (SMOTE / oversampling / undersampling rejected) to preserve temporal order
We never SMOTE, over-sample, or under-sample to balance the classes. The reason is time. The split is strictly chronological -- train is the oldest rows, then an embargo gap, then val, then test -- so the model is always tested on the future. Over-sampling copies rows; a copied minute can land in both train and test, which leaks the future into the past. SMOTE invents synthetic minutes that never happened on a real clock. Either one breaks the time order the whole pipeline is built to protect. Weighting gives the balancing effect without touching row order.

**Why this, not the alternative:** Resampling is the textbook imbalance fix, but it assumes rows are interchangeable. In a time series they are not -- their order IS the signal. Because the split is by timestamp with a session-counted embargo, any duplicate or synthetic row is a leak that backtests beautifully and loses money live. Per-row weighting achieves the same 'this class matters more' effect while every row keeps its real timestamp and stays on its own side of the split.

**Q (manager):** Everyone uses SMOTE for a 53/1 imbalance -- why don't you?  
**A:** Because our rows are minutes in time, not independent samples. The split is chronological with an embargo, so we test on the future. Oversampling would put the same minute in train and test, and SMOTE would invent minutes that never traded -- both leak the future and inflate the backtest. Weighting gets the same balancing effect without breaking time order or copying a single row.

`trainer/objective.py:84 (three_way_split, chronological); trainer/purged_cv.py:130 (embargo_end, session-counted gap); settled in CLAUDE.md tool table (SMOTE rejected)`

### Weight-aware leaf floor (XGBoost min_child_weight kept low)
XGBoost's min_child_weight is not a row count -- it is the minimum SUM of sample_weights allowed in a leaf. Our weights run from 0.00 to 0.91 and NO_TRADE is near 0, so a leaf holding 500 NO_TRADE rows can have a weight-sum near zero. If this floor is set high, XGBoost refuses to split anywhere near the low-weight and rare classes. So we keep the floor low (default 1.0) on purpose, because we are training with tiny per-row weights.

**Why this, not the alternative:** The default reasoning ('a leaf needs enough rows') is wrong once you weight rows. A high floor silently blocks the model from ever cutting into the rare, low-weight regions -- exactly the ENTRY signals we care about. Keeping it low is a direct consequence of using per-row weights that span 0 to 0.91.

**Q (manager):** Doesn't a low min_child_weight just overfit -- why keep it small?  
**A:** Here min_child_weight is a sum of weights, not a count. Since our weights go down to ~0, a big honest chunk of rare-class rows can still sum to almost nothing. A high floor would stop XGBoost splitting near those rare classes at all. We keep it low so the model can still carve out the rare ENTRY signals.

`trainer/train.py:138-142 (min_child_weight is a floor on sum of weights, keep low)`

### Macro-averaged, threshold-free evaluation (macro-F1 and per-class PR-AUC, accuracy banned)
Because 53% of rows are NO_TRADE, plain accuracy is a lie -- a model that always says NO_TRADE scores 53% and is useless. So accuracy is banned from the headline. Instead we report macro-F1 (average F1 across all 7 classes, treating the 1.2% class as equal to the 53% class), per-class PR-AUC (how well the model ranks each rare class regardless of cut-off), and a per-true-class trading cost. The trading cost computes each mistake rate PER TRUE CLASS, so the 1.2% ENTRY_SUB counts on exactly the same footing as the 53% NO_TRADE.

**Why this, not the alternative:** A single accuracy or a micro-average is dominated by the majority class and hides whether the model can find the rare, money-making signals. Macro-averaging forces every class to count equally, and PR-AUC judges the rare classes without depending on a threshold -- so a model that quietly ignores the rare classes cannot hide from the score we optimise.

**Q (manager):** What's your headline metric, and why not accuracy?  
**A:** Not accuracy -- on 53% NO_TRADE, always-say-NO_TRADE scores 53% and is worthless, so it is banned. Headline is macro-F1 plus per-class PR-AUC and a per-true-class trading cost. Macro means the 1.2% class weighs the same as the 53% class, so a model that skips the rare ENTRY signals scores badly, which is exactly what we want.

`trainer/train.py:201-265 (report_metrics: accuracy banned, macro-F1, per-class PR-AUC); trainer/objective.py:218-221 (trading_cost rate is per true class)`


---

## 3. How we score the models

### trading_cost (cost-weighted error: rate x severity)
This is the number the pipeline actually uses to pick the best model. It looks at every kind of mistake -- 'the truth was class A, the model said B'. For each mistake it multiplies two things: how OFTEN it happens, and how much it COSTS in trading terms. Then it adds all those up into one number. Lower is better. Zero would mean no mistakes. The 'how often' part is counted per true class: of all the minutes that were really class A, what fraction did the model call B? So a mistake on the rare 1.2% ENTRY_SUB weighs the same as a mistake on the 53% NO_TRADE.

**Why this, not the alternative:** Chosen over a plain error count or accuracy because in trading the mistakes are not equal. Buying big in the wrong direction can wipe the account; taking a slightly wrong size just earns less. A flat metric treats those the same. trading_cost prices each mistake by its real damage, so the model that loses the least money wins. And because the rate is per-true-class, a model cannot get a good score by ignoring the rare classes -- which is exactly why accuracy is unsafe here and this is safe.

**Q (manager):** Why optimise on a home-made cost instead of a standard metric like F1 or accuracy?  
**A:** Because trading mistakes are not equal and standard metrics pretend they are. A full reversal at max size can blow the book; a wrong size just earns less. trading_cost weights each confusion by its real trading damage times how often it happens, so the winner is the one whose mistakes cost the least money. It is also a macro metric -- the rate is computed per true class -- so a model that ignores the rare entry signals cannot hide behind it. This is the primary pick; macro_f1 and mean_pr_auc are reported next to it to show why.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/shap_logic.py:90-121 (rate x severity, rate=cnt/n_true at line 108, importance=rate*sev at line 114); summed in /home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:210-226 and /home/megaserve/Desktop/Gourav/final_pipeline/trainer/select_champion.py:58-66; winner sorted on it at select_champion.py:295`

### Severity matrix (asymmetric cost matrix over the confusion pairs)
A JSON file that gives every 'true -> predicted' pair a cost number. It is not symmetric on purpose. Two rules are baked in: (1) an unwanted trade costs more than a missed one -- a real open position can lose money, while a missed signal only loses profit you never had; (2) over-sizing costs more than under-sizing. The tiers run from 100 (full reversal at max size) down to 2-6 (right direction, wrong size). Any pair not listed falls back to a default cost of 1.

**Why this, not the alternative:** Chosen over treating all confusions equally, and over inverse-frequency weighting, because the real objective is capital preservation, not balanced accuracy. The tiers are deliberately far apart (100 vs 3) so one rare catastrophe outranks many cheap nuisances.

**Q (manager):** Where do these severity numbers come from -- are they fitted or guessed?  
**A:** They are hand-set from standard trading logic, not fitted to data. The file itself states they are a starting point and says which asymmetries they encode: unwanted trades beat missed trades, over-sizing beats under-sizing. If the trading book disagrees, you edit the file and the pipeline just obeys it. Nothing in the model is retrained to change them.

`/home/megaserve/Desktop/Gourav/final_pipeline/configs/severity_7class.json (tiers A-F, default=1); loaded at /home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:201-207 and select_champion.py:50-55`

### Macro-averaged F1
F1 for one class is the balance of its precision and its recall. Macro-F1 works out F1 for each of the 7 classes on its own, then takes a plain average. Every class counts the same -- the 1.2% ENTRY_SUB counts as much as the 53% NO_TRADE.

**Why this, not the alternative:** Chosen over weighted-F1 or accuracy, which are dominated by the big NO_TRADE class. We care about the rare signals, so we want each class to have equal say. It is reported alongside trading_cost, not used as the primary pick.

**Q (manager):** Why macro and not weighted F1?  
**A:** Weighted F1 weights each class by its size, so NO_TRADE at 53% would dominate and a model blind to the rare entries could still look fine. Macro gives every class an equal vote, which matches the fact that the rare entry signals are what make money. It is a supporting number next to trading_cost, there to show why the winner won.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:224-225 (sklearn classification_report) and train.py:247 (rep['macro avg']['f1-score'])`

### PR-AUC per class (Average Precision, one-vs-rest)
For each class the code takes the model's predicted probability for that class and measures how well those probabilities rank the true members of the class above everyone else -- across ALL thresholds, not one fixed cut-off. It is computed with sklearn's average_precision_score against a one-hot 'this class vs the rest' target, one score per class.

**Why this, not the alternative:** Chosen over precision/recall at a single threshold, which for a rare class tells you almost nothing because one arbitrary cut-off can look good or bad by luck. PR-AUC summarises ranking quality across every threshold. It is the precision-recall version rather than ROC-AUC because PR curves stay honest under heavy class imbalance.

**Q (manager):** Why PR-AUC and not ROC-AUC for the rare classes?  
**A:** ROC-AUC looks flatteringly high when a class is tiny, because it rewards ranking against the huge pool of negatives. PR-AUC -- average precision -- focuses on the positives, i.e. precision and recall, so it is not inflated by the 98% of rows that are not that class. For a 1.2% class that is the honest view.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:234-243 (one-hot at line 236, average_precision_score at line 241)`

### Mean PR-AUC (macro average of per-class Average Precision)
Take the PR-AUC of each class and average them into one number. It says, on average across all 7 classes, how well the model ranks each class -- with every class weighted equally.

**Why this, not the alternative:** Gives the scoreboard a single ranking-quality number to sit next to trading_cost, again equal-weighted per class so the rare ones are not drowned out.

**Q (manager):** What does mean_pr_auc add over macro_f1?  
**A:** macro_f1 judges the model at one fixed decision threshold; mean_pr_auc judges how well it ranks each class across all thresholds. A model can score poorly on F1 at the default cut-off but still rank well, which means a better threshold would fix it. Seeing both tells you whether the problem is the model itself or just the threshold.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:248 (np.mean over the per-class PR-AUC values)`

### Per-class precision and recall (classification_report + confusion matrix)
For each of the 7 classes the code prints two numbers. Precision: when the model said this class, how often was it right. Recall: of all the minutes that were really this class, how many did it catch. It prints the full per-class table plus a 7x7 confusion matrix showing exactly which class gets confused for which. Nothing is collapsed into one headline number.

**Why this, not the alternative:** Chosen over any single headline metric because the whole point is to see, per class, whether the model can actually find the rare ENTRY signals. A headline hides that; the per-class table and confusion matrix show it directly.

**Q (manager):** Which matters more here, precision or recall?  
**A:** For the rare entry signals, recall matters most -- a missed signal is money not made. But precision guards against over-trading, which costs real money. We report both per class and never merge them, and the confusion matrix shows which specific class is being mistaken for which, so we can see whether a loss is a reversal, an over-size, or an unwanted trade.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:224-225 (output_dict per-class precision/recall/f1) and train.py:257-261 (confusion matrix)`

### Accuracy deliberately banned (majority-class baseline argument)
Accuracy is never used to pick a model or to headline a result. The reason: NO_TRADE is 53% of the rows, so a model that always says NO_TRADE and nothing else scores 53% while being completely useless. Even 60% can mean the model never found a single entry signal. The code says so in plain words and refuses to make accuracy the headline.

**Why this, not the alternative:** Chosen against accuracy because the class balance makes it a lie here -- the dumb 'always NO_TRADE' baseline already scores 53% with zero trades. So accuracy near that number is the model doing nothing useful.

**Q (manager):** So what's your baseline, and why isn't 53% good?  
**A:** The naive baseline is 'always predict NO_TRADE'. It scores 53% accuracy, makes zero trades and zero money. Any accuracy near 53% means the model is basically doing that. That is why the code bans accuracy as the headline and picks on trading_cost, with macro_f1 and mean_pr_auc reported alongside -- metrics that only reward actually finding the rare signals.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:204-206; /home/megaserve/Desktop/Gourav/final_pipeline/trainer/select_champion.py:6-10; /home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:217-221`

### Blind-class disqualification (never_predicted hard guard)
Before naming a winner, the code lists any class the model never predicts at all. If a model is blind to a class that must be traded, it is marked unusable and cannot be tagged 'champion', no matter how good its other numbers are. Right now all three models are blind to NO_TRADE, because NO_TRADE has weight 0 in the labels, so the code refuses to crown anyone and tags the best one 'champion-BLIND' instead. Deployment only reads the 'champion' tag, so nothing picks up a blind model.

**Why this, not the alternative:** Chosen over trusting the scoreboard numbers alone, because a model that ignores a whole class can still post a respectable trading_cost and F1. Without this guard a broken model could be deployed on good-looking metrics. An earlier version of this guard was inert -- it printed a warning and then tagged the model champion anyway -- and that was fixed.

**Q (manager):** What stops a model that looks good on the metrics but never trades a whole class from going live?  
**A:** A hard guard on top of the metrics. Any class the model never predicts is recorded; if it is a class that must be traded, the model is flagged unusable and gets the tag 'champion-BLIND', not 'champion'. Production only deploys the 'champion' tag, so a blind model is never picked up. Today that guard is firing on all three models because NO_TRADE has weight 0 in the labels -- that is a label-policy fix upstream, not a model fix.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:250-251 (never_predicted); enforced in /home/megaserve/Desktop/Gourav/final_pipeline/trainer/select_champion.py:300-353`


---

## 4. Categorical encoding

### Ordinal (label / integer) encoding — hand-rolled dictionary, not sklearn's OrdinalEncoder
For XGBoost and RandomForest, every text column is turned into whole numbers. The code takes the unique text values in a column, sorts them alphabetically, and hands out 0, 1, 2, 3 in that order (GAP_DOWN=0, GAP_UP=1, NO_GAP=2). Each category becomes one integer in one column. That's it. XGBoost and RandomForest cannot read words, so the words have to become numbers first. CatBoost is skipped here because it reads words directly.

**Why this, not the alternative:** Chosen over one-hot encoding. One-hot would add a new 0/1 column per category, and the feature set is heading from 14 to 400-500 features — that explodes width and slows the trees. Trees split on a threshold ('is the code below 1.5?'), so a single integer column works fine for them; the alphabetical order is arbitrary but the tree doesn't care, it just needs a stable number per category.

**Q (manager):** Isn't ordinal encoding wrong because it invents a fake order between GAP_UP and NO_GAP?  
**A:** For linear models yes, but not for trees. A tree only ever asks 'is this code above or below a cut point', so it can carve each category out on its own; the numeric order never gets treated as a magnitude. And it saves us the hundreds of extra columns one-hot would add across 400-plus features.

`na_policy.py:180-182 (encode_categoricals), applied at trainer/train.py:404-405`

### Unknown / unseen category handling — reserved out-of-vocabulary code -1
When we apply a saved mapping to new data and hit a category the model never saw in training, we don't crash and we don't reuse an existing number. We give it -1. Real codes start at 0, so -1 is a value no real category can ever hold. The tree then isolates all the 'never seen before' rows on their own branch.

**Why this, not the alternative:** Chosen over letting it error, or filling with 0 / the most common code. Reusing a real code (say 0 = GAP_DOWN) would make a brand-new category silently masquerade as an existing one — a collision that never throws an error and quietly corrupts predictions. -1 sits outside the real range so it can't collide.

**Q (manager):** What happens live when a genuinely new market state shows up that wasn't in the training data?  
**A:** It gets encoded as -1, a code no training category ever used, so the model treats it as its own distinct 'unknown' bucket instead of pretending it's a category it already knows. There's a regression test pinning exactly this.

`na_policy.py:182 (.map(...).fillna(-1).astype('int32')), test at tests/test_na_policy.py:181-186`

### Fit-on-train-only preprocessing (leak-free encoding)
The mapping from text to numbers is LEARNED using the training rows only, then APPLIED to train, validation, and test alike. We never look at the test rows to decide the encoding. A category that only ever appears in the test period is not in the mapping, so it falls through to -1 — exactly what would happen live.

**Why this, not the alternative:** Chosen over fitting the encoder on the whole dataset at once (the lazy default). If the mapping is built from all rows, then facts about the future test period have shaped how the training data was prepared. That's data leakage: it never crashes, never shows in a metric, and quietly flatters the score. Same discipline as fit_transform on train, transform on test.

**Q (manager):** How do you know the encoding step isn't leaking test information into training?  
**A:** encode_categoricals is called once on the train slice to learn the mapping, then a second time with that fixed mapping to transform everything. Test-only categories are unknown to the mapping and become -1. There's a regression test that proves the sentinel and mapping differ when you cheat and fit on the full data.

`trainer/train.py:404-405 (with the comment at 396-401), test at tests/test_na_policy.py:190-216`

### Encoder persistence / train-serve parity — the mapping is saved with the model
The exact text-to-number mapping is stored inside the saved model file (in cat_maps). At explanation or inference time we load that same mapping instead of recomputing one. So if GAP_UP was 1 during training, it is still 1 forever after.

**Why this, not the alternative:** Chosen over recomputing the mapping from whatever data arrives at inference. If you re-derive the sorted mapping on live data, the alphabetical order can shift (a missing or extra category) and GAP_UP could become 2 instead of 1 — the model would then read the wrong feature and be silently wrong. Saving it once makes the encoding deterministic across training and production.

**Q (manager):** How do you guarantee live data is encoded the same way training data was?  
**A:** The category mapping is serialized into the model bundle under cat_maps and reloaded verbatim at inference, so the same word always maps to the same integer. Nothing is recomputed downstream — shap_explain reuses the saved mapping directly.

`trainer/train.py:497 (cat_maps saved in joblib bundle), reused at trainer/shap_explain.py:137`

### CatBoost native categorical handling (cat_features — ordered target statistics under the hood)
CatBoost never calls our encoder at all. We just make sure each text column is a string and that missing values became the label 'MISSING', then we hand CatBoost the column positions via cat_features. CatBoost turns categories into numbers itself, internally, using the target with an ordering trick.

**Why this, not the alternative:** Chosen over forcing the same blind ordinal encoding onto CatBoost. CatBoost's built-in method (ordered target statistics) is smarter than a fixed integer — it encodes each category using the target while an ordering scheme prevents that from leaking the label. Ordinal-encoding it by hand would throw away the one thing CatBoost is best at.

**Q (manager):** Why does CatBoost skip the encoding step the other two models use?  
**A:** Because native categorical support is what CatBoost is for. We pass the column indices as cat_features and it builds its own target-based encoding internally with a leak-safe ordering, so a hand-rolled integer mapping would only weaken it.

`trainer/train.py:386-392 (catboost branch) and build_model at trainer/train.py:169 (cat_features=cat_idx)`

### Missing category as an explicit 'MISSING' label (missing-indicator category)
Before any encoding, a NaN inside a text column is replaced with the literal category 'MISSING' — for all three models. A missing category is kept as its own distinct value; it is never blended into a real category like NO_GAP.

**Why this, not the alternative:** Chosen over dropping the row or filling with the most common category. A missing text value is itself a real state; folding it into NO_GAP or the mode would fabricate a signal that isn't there and let the fake values swamp the true ones. Giving it its own label keeps the states separate.

**Q (manager):** What do you do with a missing value in a text feature — drop it or guess it?  
**A:** Neither. We turn it into its own category called MISSING so the model can learn that 'we don't know' is itself informative, rather than silently merging it into a real category and inventing a signal.

`na_policy.py:94-98 (apply_policy, cat_cols branch) and trainer/train.py:390; label set in config.py:294 (CATEGORICAL_NA_LABEL)`

### Target label encoding via sklearn LabelEncoder (the 7 class strings)
Separately from the feature encoding, the 7 class label strings (NO_TRADE, ENTRY_SUPER, ...) are turned into 0-6 with sklearn's LabelEncoder. The fitted encoder is saved with the model, and its class list is reused to make sure predicted-probability columns line up with the right class.

**Why this, not the alternative:** Chosen over a manual dict for the target. LabelEncoder also exposes classes_, which the code needs to rebuild a full 7-column probability matrix when RandomForest omits a class it never saw in a split — a hand-rolled dict wouldn't give that safety net for free.

**Q (manager):** How do the 7 text labels become something the model can train on, and how do you keep the probability columns aligned?  
**A:** sklearn's LabelEncoder maps the 7 strings to 0-6 and is saved with the model; its classes_ list is used by full_proba to place each model's probabilities into the correct class column, so a short RandomForest split that misses a class can't misalign the outputs.

`trainer/train.py:379-381 (LabelEncoder().fit), used by full_proba at trainer/train.py:178-198`


---

## 5. Missing values

### Sentinel-value imputation (out-of-range sentinel)
A missing number gets filled with a value the feature can never really have. A tree then makes one cut and all the missing rows fall onto their own branch. Because the fill value is outside the real range, it can never be mistaken for a real number.

**Why this, not the alternative:** The obvious alternative is fill with 0 or the column mean. Both are values the feature CAN take, so 'missing' would land on the same number as real rows and the model could not tell them apart. Example in the code: gap_fill_ratio=0 already means 'gap fully open, nothing filled' -- a real state -- so filling NaN with 0 makes 86% of no-gap minutes look like that state and buries the signal.

**Q (manager):** Why not just fill missing values with 0 or the column mean?  
**A:** Because 0 and the mean are real, reachable values for the feature. gap_fill_ratio=0 means 'there is a gap and none of it filled' -- a genuine tradeable state. Fill NaN with 0 and 86% of no-gap minutes carry that same number, so two different meanings collide and the model ignores it. A sentinel sits outside every real value, so it can never collide.

`final_pipeline/na_policy.py:8-31 (rationale), 137-161 (apply); config.py:266-273`

### Per-column computed sentinel (compute_sentinel formula)
For each column we place the sentinel a full range below the smallest real value, and then one margin lower still. Formula: sentinel = min - (max - min) - 1. That guarantees no real row, in whatever units the feature uses, can ever land on it.

**Why this, not the alternative:** A single hardcoded number is not safe for every column. A feature measured in points could legitimately BE -999, which would recreate the exact collision we are avoiding. The formula adapts to each column so it is always out of range, and the exact value used is written into the manifest so it is never a mystery later.

**Q (manager):** How do you know your sentinel is actually out of range for that specific column?  
**A:** We do not hardcode it. compute_sentinel does min - (max-min) - 1 per column, which is provably below every value that column actually holds, whatever its units. The value it picked is recorded in the manifest so anyone can check it afterwards.

`final_pipeline/na_policy.py:48-62; config.py:282-288 (SENTINEL_MARGIN=1.0)`

### Flat fixed sentinel -999 (current project decision, overrides the formula)
Right now the project turns the per-column formula OFF and fills every numeric NaN with one flat -999 for all three models. No per-column math, and no NaN is left anywhere -- so even xgboost and catboost see -999 instead of a real NaN. It is a single switch: set NA_FIXED_SENTINEL back to None and the per-column formula returns, nothing else changes.

**Why this, not the alternative:** The per-column formula is the safer design (it can never collide), but a flat -999 is one number that is easier to explain and audit line by line to a manager, and it is already out of range for the current feature set so the collision the formula guards against cannot happen here. That trade-off -- simpler to explain, at the cost of a here-impossible collision -- is stated in the code.

**Q (manager):** You said the per-column formula is safer -- so why are you shipping the flat -999 instead?  
**A:** Because for this exact feature set -999 is already outside every column's real range, so the collision the formula protects against cannot occur here, and one flat value is far easier to explain and audit. It is not permanent -- NA_FIXED_SENTINEL=None restores the per-column formula with no other code change.

`config.py:306-311 (NA_FIXED_SENTINEL=-999.0); na_policy.py:138-146; bridge/build_dataset.py:434-452`

### Native missing-value handling (sparsity-aware split finding)
xgboost and catboost can train on real NaN directly -- they learn, per split, which way to send a missing value. So when the flat -999 is OFF, we leave their NaN untouched instead of filling it. Only random forest, which crashes on NaN, is given a sentinel.

**Why this, not the alternative:** Filling the NaN would throw away the fact that these models handle it better themselves. They pick the best branch for 'missing' straight from the data, rather than us guessing a fill value. We only fill when we must (random forest, or when the flat -999 switch is on so all three share one feature list and the parquet has no NaN).

**Q (manager):** If xgboost and catboost handle NaN on their own, why fill it at all?  
**A:** With the flat -999 switch on we fill for all three so they share one feature list and the parquet carries no NaN. With it off, xgboost and catboost keep the real NaN and learn a branch for it (sparsity-aware split finding); only random forest gets a sentinel because it cannot train on NaN.

`final_pipeline/na_policy.py:30-32, 147-152; bridge/build_dataset.py:264-275`

### Explicit missing category ("MISSING" as its own level)
For text and category columns, a missing value becomes its own label, the literal string MISSING. It is never blended into a real category. This is the same idea as the numeric sentinel, but for text -- a number cannot live in a text column.

**Why this, not the alternative:** If a missing gap_state silently became NO_GAP, two different meanings would share one label and the model could not tell 'we do not know' from a real state. Keeping MISSING as a separate category preserves that distinction.

**Q (manager):** What happens to a missing text or category value?  
**A:** It becomes the distinct category 'MISSING'. It is never merged into a real category like NO_GAP, so the model can still separate 'unknown' from a genuine state.

`final_pipeline/na_policy.py:90-98; config.py:293-294 (CATEGORICAL_NA_LABEL); bridge/build_dataset.py:446-450; trainer/train.py:390`

### Bounded forward-fill (LOCF with a gap limit)
For the 'ffill' policy, a slow value is carried forward onto the faster rows -- but only for a few bars. Past the cap the row is left NaN. The cap is limit = bar_minutes x 3.

**Why this, not the alternative:** An unbounded forward-fill would carry yesterday's last value across the overnight gap and hand it to this morning as a fresh feature -- a stale number pretending to be current. The cap stops the value leaping the gap.

**Q (manager):** Doesn't forward-fill risk dragging a stale value across the overnight gap?  
**A:** That is exactly why it is bounded. The fill is capped at bar_minutes x 3 bars, so a value can only be carried a few bars; beyond that it stays NaN rather than becoming tomorrow morning's feature.

`final_pipeline/na_policy.py:110-119; config.py:290-291 (STALE_TOLERANCE_BARS=3)`

### Drop-as-mask (leak-safe row exclusion)
The 'drop' policy does not delete the bad rows on the spot. It hands back a mask -- the list of unusable timestamps -- and build_dataset removes those label minutes later.

**Why this, not the alternative:** If we deleted the feature rows here, alignment would forward-fill over the hole and quietly serve the model a stale value, so 'drop' would secretly behave like 'ffill' -- the opposite of dropping. Keeping the row and returning a mask makes the exclusion real. Measured: a 3-minute hole handed the model a value from 4 minutes earlier and nothing complained.

**Q (manager):** Your 'drop' policy keeps the row -- so what is it actually dropping?  
**A:** It drops the label minute, not the feature row. If we deleted the feature row, alignment would fill the hole with a stale value and the drop would silently turn into a forward-fill. So we keep the row, return a mask of the bad timestamps, and the label minutes themselves are removed downstream.

`final_pipeline/na_policy.py:121-135`

### Fit-on-train-only imputation (leakage control)
When random forest needs a sentinel, that sentinel is computed from the training rows only, then applied to the test rows. The category-to-number mapping is learned the same way -- train only, applied to everything.

**Why this, not the alternative:** If the sentinel or the mapping were computed from the whole dataset, facts about the test period would shape how the training data was prepared. That is a leak that never crashes and never shows in a metric, but quietly flatters the score. Fitting on train and applying to test mirrors live, where the future does not exist yet.

**Q (manager):** Do you compute the sentinel and the category codes on all the data or just the training slice?  
**A:** Training slice only. compute_sentinel and the category mapping are fit on the train rows and then applied to test, so no test-period information touches data preparation. It matches production, where you only ever have the past.

`final_pipeline/trainer/train.py:396-420 (compute_sentinel on X.loc[tr]); na_policy.py:166-183 (encode_categoricals, unseen category -> -1)`


---

## 6. No-lookahead time alignment

### Point-in-time as-of join on bar close (pandas merge_asof, direction=backward)
Every label minute gets the newest feature bar that has already finished. A bar stamped 09:15 on a 5-minute clock is not done until 09:20, so no minute before 09:20 may see it. We compute each bar's close time, then join each minute to the last bar whose close is at or before that minute.

**Why this, not the alternative:** The obvious fix was shift(1) -- move the feature back one row. But a 5-minute value is copied across five 1-minute rows, so one row back is one MINUTE back, not one BAR back. shift(1) hides only one of the five leaking minutes. Joining on the real close time fixes every clock at once.

**Q (manager):** How do you know a minute never sees a bar that hasn't closed yet?  
**A:** merge_asof with direction=backward can only return a bar whose close is <= the minute. A separate tripwire assert re-checks every row and raises NoPeekViolation if any served close is later than the minute -- it catches anyone later 'fixing' the direction= flag.

`bridge/align.py:404-410 (the join), 327-330 (bar_close), 424-427 (NoPeekViolation assert)`

### Session-anchored bar boundaries (anchor at 09:15 = 555 minutes, not floor-from-midnight)
To know when a bar closes we first find where it starts. The market opens at 09:15, which is 555 minutes after midnight. We measure every bar boundary from 09:15, not from midnight.

**Why this, not the alternative:** pandas floor() counts from midnight, and 555 does not divide by 30 or 60. A 30-min session bar 09:15-09:44 (really closes 09:45) floors to 09:00 and looks like it closes at 09:30 -- served 15 minutes early, a silent leak. Anchoring at 555 puts the boundary in the right place. For 1/3/5/15 min it changes nothing because they all divide 555, so this only matters when we start using 30/60-min features.

**Q (manager):** Why not just use pandas floor to bucket the bars?  
**A:** floor counts from midnight and 555 isn't divisible by 30 or 60, so 30- and 60-minute bars would appear to close 15 minutes early -- a leak the no-peek assert can't see because the arithmetic is self-consistent. Anchoring at the session open makes the close correct for every clock.

`bridge/align.py:303-324 (_bar_start), config.py:155 (SESSION_ANCHOR_MINUTES = 555)`

### Empirical clock detection -- measure each column's real bar period instead of trusting the declaration
We do not trust the registry's clock: field. For every column we measure how often it actually changes and infer its true bar period from the data itself. That measured period is what drives the alignment.

**Why this, not the alternative:** registry.yaml declared 5min for all 14 features, but measured, several are really 1min. A declaration that is too fast is an undetectable leak. Worse, register.py WRITES the clock: field from this same measurement, so trusting it would be comparing the measurement against a copy of itself -- a guard that can never fire. So build_dataset always uses the slower of (declared, measured); going faster needs a named human override recorded in the manifest.

**Q (manager):** The feature team already tells you the clock in registry.yaml. Why re-measure it?  
**A:** Because a wrong-too-fast declaration is a silent lookahead leak, and that declared value was itself auto-written from a measurement, so trusting it checks nothing. We measure independently and always take the slower of declared vs measured; the only way to override is an explicit human claim that gets shouted on screen and written to the manifest.

`bridge/align.py:156-194 (clock_of_column), 226-251 (column_clocks); bridge/build_dataset.py:252, 289-294, 309-315`

### Slowest-fit clock rule -- the LARGEST period the column never moves inside
A column's clock is the largest candidate period (from 1,3,5,15,30,60) during which the value never changes inside a bar -- not the smallest. If a value holds still across every 15-min bar it automatically holds still inside every 3-min bucket too, so the old smallest-fit rule wrongly called almost everything 3min and served those bars up to 57 minutes early.

**Why this, not the alternative:** The two error directions are not symmetric. Too-long clock = value served late = stale = costs a little signal, but honest. Too-short clock = bar served before it closed = lookahead = backtests beautifully and loses money live. When the data is ambiguous we deliberately round toward the slow, safe answer.

**Q (manager):** If several bar sizes fit the data, which one do you pick and why?  
**A:** The largest that fits. Erring slow only makes a value stale; erring fast serves a bar before it closed, which is lookahead. Those costs aren't equal, so every tie resolves toward slow/safe. A column that just rarely changes will look slow -- that's safe, and it's why we hand the human the change-evidence, not only the number.

`bridge/align.py:185-194 (best keeps the largest, loop does not break), 42-44 and 169-174 (the asymmetry rationale), 283-300 (infer_bar_minutes takes max)`

### Per-column clock alignment -- group columns by clock, align each group on its own bar-close
One parquet can hold columns on different clocks -- a 5-minute signal sitting next to the 1-minute price it was built from. We measure each column separately, group the columns by their clock, and run the no-peek join once per clock group.

**Why this, not the alternative:** Forcing one clock on the whole file is wrong both ways: the fastest clock serves the 5-minute column before it closes (lookahead); the slowest clock holds the 1-minute column back (needlessly stale). Column-by-column keeps each one honest without dragging the fast columns down.

**Q (manager):** What if a single feature file mixes fast and slow columns?  
**A:** Each column is aligned on its own measured clock. We bucket columns by clock and align each bucket separately, so a 1-min price and a 5-min signal in the same file are both served correctly -- neither leaked nor needlessly stale.

`bridge/build_dataset.py:324-335 (group by clock, concat back in original order); bridge/align.py:226-241 (column_clocks rationale)`

### Staleness tolerance / forward-fill cap (tolerance_bars, then NaN)
A closed bar is allowed to carry forward only a few bars (default 3). Past that limit the value becomes NaN. This stops Friday 15:25's value from filling across the weekend and landing on Monday 09:15.

**Why this, not the alternative:** Without a cap, a feature that stops updating fills a constant forever and jumps gaps. NaN at the start of a new session is the honest answer -- no bar has closed yet. The cap is measured in BARS, not rows, so it scales with each column's clock; it was once hardwired to 1, which capped a 5-min feature at 3 source rows instead of 3 bars.

**Q (manager):** A feature stops printing for a while -- do you keep repeating its last value?  
**A:** Only for tolerance_bars (3 by default). After that it goes NaN rather than pretending a stale value is current. That also blocks weekend and overnight gap leaps, where a Friday value would otherwise leap onto Monday's open.

`bridge/align.py:337-344 (docstring), 378 and 409 (tolerance = bar_minutes * tolerance_bars)`

### Fail-closed on timezone-aware timestamps (refuse, don't strip)
If a feature parquet or the label spine arrives with timezone-aware timestamps, we stop with an error instead of quietly stripping the zone. The labels are naive IST and everything must match that.

**Why this, not the alternative:** The old code called tz_localize(None), which keeps the wall clock. A UTC parquet from the feature team's Windows box turns 09:15 UTC into naive 09:15 and matches label minute 09:15 IST -- a silent 5.5-HOUR lookahead that merge_asof, the staleness tolerance, and the NoPeekViolation assert all wave through, while the manifest still swears no_peek.applied=true. There is no safe way to guess the intended zone, so a human must convert it.

**Q (manager):** What stops a timezone mistake from silently leaking 5.5 hours of future?  
**A:** We refuse tz-aware timestamps outright and raise TimezoneAwareFeature -- on both the feature side and the label side. Dropping the zone keeps the wall clock and creates an undetectable 5.5-hour lookahead, so we force a human to convert to naive IST before the join.

`bridge/align.py:81-111 (_clean_index, feature side), 371-376 (label side)`

### Relative-tolerance constancy test, fail-slow on NaN and bool
To decide whether a column moves inside a bar, we measure its spread and compare against a tolerance scaled to the column's own magnitude -- not exact equality. Bool columns are tested by category (did it change or not). Missing values count as unknown, never as movement.

**Why this, not the alternative:** Exact equality (nunique) breaks on float32 round-trip jitter: a 5-min value that wobbles in the 12th decimal reads as a 1-min column -- the fast, leak direction. Counting a NaN as a distinct value also made a slow column with one missing minute measure as 1-min. Both mistakes point toward lookahead, so the test is deliberately built to err slow. Bool also can't be subtracted in numpy, so ~30 bool columns had been crashing the whole file's registration.

**Q (manager):** Float noise makes a slow feature look like it changes every minute -- how do you avoid labeling it fast?  
**A:** We compare the in-bar spread against a tolerance relative to the column's own size, so 12th-decimal jitter doesn't count as movement, and we treat NaN as unknown rather than as a change. Every ambiguous case resolves toward the slower, safe clock; bool columns use a plain did-it-change test.

`bridge/align.py:114-153 (_is_constant_within: bool branch 137-141, relative-tolerance numeric branch 143-151)`


---

## 7. The leak guard

### Lookahead name denylist (forward-return regex block)
Before any math we look at the column name. If the name says it points forward in time (fwd_, forward_, next_, ahead_, signed_ret, fut_ret) we refuse it. Those columns are returns measured over the NEXT few minutes. That is the answer, not a feature. It is free to run and it catches the honest cases.

**Why this, not the alternative:** A blunt 'contains fut' rule was too greedy. It wrongly killed fut_spot_spread (the futures-minus-spot basis) and open-interest change columns, which are real backward-looking inputs. So we ban only genuine forward-RETURN names and let the measured test catch anything sneaky. Chosen over a wider name ban because a guard that deletes good features gets switched off.

**Q (manager):** Why not just ban every column whose name starts with 'fut'?  
**A:** Because in this project 'fut' means the futures instrument (basis, open interest), not future-in-time. Banning it dropped documented, backward-looking inputs from the dataset. We ban fwd/forward/fut_ret only, and the correlation test below catches the rest.

`bridge/leak_guard.py:78-82 (patterns), applied at 213-214`

### Target-leakage name ban, hard-block plus soft-warn (BANNED vs SUSPECT)
We hard-block names that are clearly the label: label_int, target, y, primary_label, weight, and anything ending _at_label. But the word 'label' is ambiguous here. label_combined is a real feature (identical to Stress_Signal) and Flow_State_Label is just a state name. So a precise list is a hard block, and a looser 'smells like a label' list only prints a loud warning for a human to check.

**Why this, not the alternative:** A guard that silently deletes real features gets turned off, and then it guards nothing. Splitting into a hard ban and a soft SUSPECT keeps it precise enough that people trust it and leave it armed. Chosen over one broad ban because one broad ban already killed two good features once.

**Q (manager):** How do you avoid dropping a real feature that just happens to have 'label' in its name?  
**A:** The hard-ban list is exact matches only. Anything that merely contains 'label' or 'target' becomes a SUSPECT: printed, not dropped. A human then either clears it via allow_columns or removes it.

`bridge/leak_guard.py:94-100 (label + soft-label patterns), 216-225`

### Behavioural leak test: forward-return vs past-return correlation
Names lie, so we also measure. For every numeric column we build the return over the NEXT 1, 5, 10 minutes and the return over the LAST 1, 5, 10 minutes, then correlate the column with both. A real 1-minute signal correlates about 0.02 to 0.05 with the future. If a column tracks the future strongly, and tracks the future far more than the past, it knows the answer.

**Why this, not the alternative:** This is the layer that catches the NEXT leak, the one named 'score' that nobody looks at twice. There are three cutoffs: above 0.30 with the future is banned outright, above 0.10 AND at least twice its past-correlation is banned, and a milder future lean is only flagged. The past comparison exists because momentum features are built from past returns so they correlate with the past by design; the answer does not.

**Q (manager):** Why compare future correlation to past correlation instead of using one flat threshold?  
**A:** A momentum feature is made of past returns, so it correlates with the past on purpose. A leaked label correlates with the future and not the past. Requiring the future to beat the past by 2x separates real edge from the answer, and a hard 0.30 ceiling catches columns that correlate with both.

`bridge/leak_guard.py:187-188 (future/past returns), 265-296 (test), thresholds 106-112`

### Calendar / identifier memorisation guard (date, monotonic clock, running-ID)
A date or a row id is not lookahead, but it is bad in another way: the model memorises which day each row is, scores great on train, then meets test rows whose ids it never saw. We catch this four ways: a datetime column, a date written as text like 2024-01-05, a number that climbs almost perfectly with row order (corr above 0.97), and a running id counter that mostly sits at its own running maximum.

**Why this, not the alternative:** The two obvious checks miss real ids. A running event id has a -1 'no event' sentinel mixed in, so it is not monotonic and its time-correlation can be as low as 0.17. Its true fingerprint is that its value is almost always the newest id seen so far. Measured on the real files episode_id scored 1.000 on that test and every honest column scored under 0.013.

**Q (manager):** A counter isn't a price and can't see the future, so why ban it?  
**A:** Because every id in the test period is one training never saw, so the model just memorises which era each id belongs to. It looks exactly like overfitting and is very hard to find. We fingerprint it by how often the value sits at its own running maximum.

`bridge/leak_guard.py:200-210 (datetime, date-as-text, name), 237-242 (monotonic time corr), 252-263 (running-id fingerprint)`

### Report-only (shadow) mode via feature flag: LEAK_GUARD_ENFORCE
The guard has two modes set by one switch. When LEAK_GUARD_ENFORCE is True it drops every flagged column. When False it still runs and still prints everything it WOULD drop, but keeps the columns. Today it is False because the feature team confirmed the current feature set has no lookahead. The findings are logged, the columns stay.

**Why this, not the alternative:** Enforcing on a false alarm deletes a real feature, and then people turn the guard off for good. Report-only lets the guard keep watching and logging while a human owns the call, instead of silently mangling the dataset. Chosen over hard-enforce so the guard stays on and trusted; flip the flag back to True to re-arm.

**Q (manager):** If the guard is in report-only mode, what actually protects the dataset today?  
**A:** Two things still bite even in report-only: the raw calendar columns are always dropped, and every flag is written to the log for review. Only the correlation-based drop is paused, on the feature team's written sign-off, and re-arming is a one-line flag change.

`config.py:296-300, enforced/branched at bridge/build_dataset.py:219-228`

### Unconditional calendar drop (always-drop, overrides mode and allowlist)
Some columns are never a feature in any mode: session, t5, date, datetime, timestamp, expiry_date. These are dropped even in report-only mode, and even an 'allow everything' override cannot put them back. A raw date is pure calendar memorisation with no upside.

**Why this, not the alternative:** Report-only keeps flagged columns for a human to review, but a raw timestamp has no legitimate use as a feature at all, so there is nothing to review. Dropping it before the mode check and before the allowlist means no mode and no override can ever leak the calendar back in.

**Q (manager):** You said report-only keeps flagged columns, so could a raw timestamp still slip through?  
**A:** No. Calendar columns are on a separate always-drop list that runs before the mode branch and before the allowlist. Report-only keeps everything else; the raw date is gone regardless of mode.

`config.py:301-304 (CALENDAR_ALWAYS_DROP), applied at bridge/build_dataset.py:216, 220, 224`

### Auditable allowlist override (allow_columns escape hatch)
A human who has actually looked at a column can force it through by listing it under allow_columns in registry.yaml with a comment saying why. The guard then skips that column on every future scan. It is on the record, in the file, with a name against it.

**Why this, not the alternative:** Overrides are needed because label_combined really is a good feature. But a hidden override is the same as no guard. Forcing the exception into a version-controlled YAML with a reason keeps it visible and accountable, instead of someone quietly disabling the whole guard.

**Q (manager):** Doesn't an override just let people bypass the guard whenever it is inconvenient?  
**A:** It is per-column, written in the registry, and reviewed like any code change, so no one can override in secret. The alternative, people turning the whole guard off, is far worse.

`bridge/leak_guard.py:139 (allow arg), 196-197 (skip), doc 153-158`

### Defence in depth: screen the parquet, not just the registry (double screen)
The same guard runs twice. Once in register.py when a feature goes on the ballot, so a leak is never even offered to the expert. Again in build_dataset.py, which screens the actual parquet we ingest, not the registry's list of column names.

**Why this, not the alternative:** A ban that only lives in registry.yaml is decoration. Someone can hand-edit the YAML to add a banned column back, or the feature team can quietly re-drop a file with a new column and nobody re-runs register. The parquet is what we actually train on, so the parquet is what we screen.

**Q (manager):** Why screen twice, isn't the registry check enough?  
**A:** The registry is a list of names; the parquet is the real data. If the file changes after registration, or the YAML is edited by hand, only the second screen on the parquet catches it. The build step reads the exact file we load.

`bridge/register.py:116-119 (ballot screen), bridge/build_dataset.py:206-213 (parquet re-screen)`

### Deterministic categorical encoding (sorted mapping, not factorize)
To correlate a text column against price we turn its categories into numbers. We sort the categories and number them in sorted order. We do not use pandas factorize, which numbers categories by whichever row shows up first.

**Why this, not the alternative:** factorize gives the same column different codes when the file is in a different row order, so a correlation could appear or disappear just from ordering. Sorting makes the encoding a property of the data itself, so the leak test returns the same verdict on any machine and any file order.

**Q (manager):** Why does the encoding method matter for a leak test?  
**A:** Order-of-appearance encoding is not reproducible: reorder the rows and the numbers change, and so can the correlation. Sorted encoding is stable, so the guard's decision never depends on how the file happened to be sorted.

`bridge/leak_guard.py:123-130 (_numify sorted mapping)`


---

## 8. Train / validation / test split & cross-validation

### Time-based (chronological) train/val/test hold-out split -- no shuffling
The rows are cut by the clock, not at random. Oldest data trains the model, the middle slice tunes it, and the newest 30% is the test set. The cut points are quantiles of the timestamp, so it is always the same rows in time order no matter how the file is sorted. Nothing is shuffled, ever.

**Why this, not the alternative:** The obvious default is sklearn train_test_split with shuffle=True or plain KFold. Both scatter future minutes into the training set and past minutes into the test set. Two minutes one apart share almost all of their 20-day rolling history, so a shuffled split scores the model on rows it has effectively already seen. The number looks great and means nothing. Time order is the only honest cut for a live intraday signal.

**Q (manager):** Why not just do a normal 80/20 random split like everyone else?  
**A:** Because a random split lets the model peek at the future. Row 09:31 and 09:32 carry almost the same rolling-window inputs, so if one is in train and the other in test the model is scored on data it already memorised. We train on the old period and test on the most recent 30% only, so the test measures what live trading will actually see.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:84-161 (three_way_split, cuts via ts.quantile at :146 and :158); /home/megaserve/Desktop/Gourav/final_pipeline/config.py:102 (TEST_FRACTION=0.30) and :115 (VAL_FRACTION=0.15); called at /home/megaserve/Desktop/Gourav/final_pipeline/trainer/train.py:349-351`

### Separate validation set for tuning; test opened once (nested hold-out, guards against selection bias)
There are three slices, not two. The hyper-parameter search is only ever allowed to look at the middle validation slice. The test slice is locked until the very end and is opened one time, on the single winning model. The tuner reads a scalar called val/trading_cost and never test/trading_cost.

**Why this, not the alternative:** The obvious shortcut is to tune on the test set and report the test score. But the moment you run 30 trials and keep the best test number, the test set has entered the training loop -- you fitted the settings on it, and settings are parameters too. The winner is then the best of 30 noisy draws, biased low by construction. The code works a concrete example: with identical models and noise spread 1.5, best-of-50 lands ~2 std below the true mean, about 3 points of trading_cost that do not exist. You would report 38, deploy it, and live-trade a 41.

**Q (manager):** If you already have a test set, why do you need a third validation set -- isn't that wasting data?  
**A:** Because tuning on the test set quietly corrupts it. Picking the best of many trials makes the winning score too good by luck, so there would be nothing honest left to report. We tune on validation and touch test exactly once on the final model. That test number is the only one we say out loud to you.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:88-105 (the worked example) and :79 (OBJECTIVE_SERIES = val/trading_cost); /home/megaserve/Desktop/Gourav/final_pipeline/config.py:98-114; /home/megaserve/Desktop/Gourav/final_pipeline/trainer/hpo.py:242 ("the test set is not touched") and :249-251`

### Embargo / purge gap between slices, counted in TRADING SESSIONS (not calendar days, not rows)
Between train and val, and again between val and test, a gap of 25 trading sessions is thrown away. This stops the last training rows and the first test rows from sharing rolling-window history. The gap is measured in market sessions read straight off the timestamps -- not calendar days and not row counts. Two embargoes, one before each cut.

**Why this, not the alternative:** The old code used cut + 21 calendar days. That is wrong: a feature like ret_20d looks back 20 trading sessions, but 21 calendar days only spans about 14 sessions because weekends and holidays are shut -- measured on the real file: mean 14.1, min 9, max 16, never 20. So the gap was ~30% short and the first ~6 sessions of every test set still carried features built from training prices. Counting rows is also wrong: sessions are not all 375 rows (half-days exist), so 20*375 rows can land you inside the window you meant to skip. Sessions are the unit the features are actually built in.

**Q (manager):** You throw away 25 days between each slice -- why sessions instead of just adding 21 days like before?  
**A:** Because 21 calendar days is only about 14 open-market days, and our features look back 20 open-market days. The old day-based gap covered three-quarters of the lookback, so the leak survived. We now count the same thing the feature counts -- trading sessions off the real calendar -- so 25 sessions fully clears the 20-session lookback plus margin. There is a test that proves 21 calendar days never reaches 20 sessions anywhere in five years.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/purged_cv.py:130-166 (embargo_end); /home/megaserve/Desktop/Gourav/final_pipeline/config.py:135 (EMBARGO_SESSIONS=25); the two embargoes at /home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:107-114,147,159; proven by /home/megaserve/Desktop/Gourav/final_pipeline/tests/test_purged_cv.py:118 (test_calendar_days_are_NOT_trading_sessions)`

### Purged & embargoed k-fold cross-validation, asymmetric cuts (Lopez de Prado combinatorial-purged CV family) -- BUILT AND TESTED, not yet wired into the live trainer
This is a k-fold splitter that respects time. Around each test fold it removes training rows on BOTH sides, but by different amounts. Before the fold it purges H sessions (H = how far a label looks forward). After the fold it embargoes H+L sessions (L = how far a feature looks back, plus the fold's own labels reaching forward). It cuts on whole session boundaries, never mid-day. Important honesty point: this class is fully written and has 12 passing tests, but the production trainer does NOT call it yet -- train.py and hpo.py use the three-way time hold-out above. Only the test files import PurgedKFold.

**Why this, not the alternative:** sklearn TimeSeriesSplit is the obvious choice but it has no purge and no embargo (grep the package for 'purg' -- zero hits), and it only ever trains on data before the fold, throwing away everything after. The other common move is to subclass sklearn's BaseCrossValidator and define the test folds -- but that base class sets train = every row not in test, so it silently applies no purge at all while still being called PurgedKFold. This class owns split() outright and inherits nothing, and the cuts are asymmetric on purpose: a symmetric gap either wastes data on the left or leaks on the right.

**Q (manager):** Is this purged k-fold actually running when we train, or is it just sitting there?  
**A:** Right now it is built and tested but not wired into training -- the live trainer uses the single time-ordered validation hold-out. The k-fold splitter is ready for when we want a fuller cross-validated estimate; its asymmetric purge/embargo and its leak checks all pass 12 tests. Turning it on in the trainer is a deliberate next step, not an accident.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/purged_cv.py:169-267 (PurgedKFold, asymmetric cuts at :242-246); tests at /home/megaserve/Desktop/Gourav/final_pipeline/tests/test_purged_cv.py:87 (asymmetry), :172 (TimeSeriesSplit trap), :205 (subclass trap); NOT imported by trainer -- /home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:133 pulls only embargo_end, and PurgedKFold has 0 hits in train.py/hpo.py`

### Post-split leakage assertion (no-overlap check + purge/embargo guard)
After the slices are made, the code goes back and proves there is no leak on the real rows it is about to hand the model. three_way_split counts how many rows land in two slices at once and refuses if any do. The k-fold path has assert_no_leak, which re-checks in session distance that no training row before a fold is within the purge, and none after is within the embargo, and that no row is in both train and test.

**Why this, not the alternative:** The purge and the embargo are pure bookkeeping. If you get the arithmetic wrong nothing crashes, nothing looks odd, the model just scores better than it deserves. A silent leak is the worst kind. So instead of trusting the split logic, the code independently verifies the property it claims to have, on the actual arrays -- a guard that fires on a deliberately broken split (proven by a test that feeds it a naive no-gap split and demands PURGE FAILED).

**Q (manager):** How do you actually KNOW there is no leak -- not just that you intended one?  
**A:** We check it after the fact on the exact rows going into the model. The split raises if any row is in two slices, and the k-fold guard re-measures the gap in sessions and raises if any training row sits inside the purge or embargo. We have a test that hands the guard a broken split with no gap and confirms it throws -- a guard that never fires would be useless.

`/home/megaserve/Desktop/Gourav/final_pipeline/trainer/objective.py:178-181 (overlap check in three_way_split); /home/megaserve/Desktop/Gourav/final_pipeline/trainer/purged_cv.py:291-328 (assert_no_leak); proven by /home/megaserve/Desktop/Gourav/final_pipeline/tests/test_purged_cv.py:273 (test_assert_no_leak_catches_a_deliberately_broken_split)`


---

## 9. Hyperparameter optimisation (HPO)

### Random Search (ClearML RandomSearch) — the actual HPO algorithm
The search picks each trial's settings at random from the allowed ranges, trains a model, keeps the best. It does NOT try every combination. Run 30 trials over 5 knobs and you test 30 different values of every knob.

**Why this, not the alternative:** Grid search wastes its budget re-testing knobs that don't matter: a 3x3x3x3x3 grid is 243 runs but only 3 values per knob, while 30 random draws explore 30 values of each. With 4+ knobs and only 1-3 agents, random wins for a fixed budget. Optuna and BOHB (the smarter Bayesian options) are not installed on the box and we chose not to install them.

**Q (manager):** What search algorithm is it — grid, random, or Bayesian? And why?  
**A:** Random search, ClearML's RandomSearch class. It beats grid for a fixed compute budget once you have 4+ knobs, and it's plain to explain in a meeting. Bayesian (Optuna/BOHB) needs extra packages that aren't on the machine and isn't worth it for 3 models and a handful of knobs.

`trainer/hpo.py:186 (--strategy default "random"), trainer/hpo.py:204 (maps to RandomSearch); reasoning trainer/hpo.py:71-88`

### Grid Search (ClearML GridSearch) — the optional alternate strategy
A second mode you switch on with --strategy grid. It tries every combination of a small hand-picked set of values, e.g. 4 depths x 3 leaf sizes = 12 runs.

**Why this, not the alternative:** Kept only for a small, deliberate sweep you want to walk through line by line in a meeting, where seeing every rung matters more than covering ground. For a real search it loses to random, so random is the default.

**Q (manager):** So you can do grid search too — when would you use it over random?  
**A:** Only for a small explainable sweep, like max_depth in {0,16,24,32} x min_samples_leaf in {5,25,100}, 12 runs, where I want to show every point. For anything real I use random because grid burns budget on knobs that don't matter.

`trainer/hpo.py:186,204 (strategy switch); note trainer/hpo.py:86-88`

### ClearML HyperParameterOptimizer — clone-enqueue-poll loop (not a callback)
The optimizer takes one base training task, clones it, overwrites a few settings, and drops the clone on the queue. An agent runs it as a normal training job. When it finishes, the optimizer reads ONE number off it and picks the next settings. There is no scoring function of ours that it calls.

**Why this, not the alternative:** Reusing the exact train.py an engineer runs by hand means tuning and manual training can never drift apart, and the trials ride the same agents/queues we already have. A custom score-function loop would be more code and wouldn't reuse the fleet.

**Q (manager):** How does the optimizer get each trial's score if it never calls your code?  
**A:** It reads one scalar, Summary/val/trading_cost, off the finished ClearML task via last_metrics. If that string doesn't match, ClearML returns None silently and every trial ties, so we pin the exact name in code and in a test.

`trainer/hpo.py:246-270 (optimizer build); loop explained trainer/hpo.py:36-68; objective strings trainer/objective.py:78-80`

### Custom objective — validation trading_cost, minimized (a macro cost, not accuracy)
We do not tune for accuracy. We tune to make trading_cost as LOW as possible. trading_cost = sum of mistake-rate times how bad each mistake is, computed per true class. Lower is better.

**Why this, not the alternative:** Accuracy is a lie here: 53% of rows are NO_TRADE, so a model that never trades scores 53%. trading_cost is macro, so the 1.2% ENTRY_SUB counts as much as NO_TRADE, and a full direction reversal is punished far harder than a wrong position size. Optimizing accuracy would breed a model that never trades.

**Q (manager):** What exactly is the search optimizing, and can it be gamed by ignoring the rare classes?  
**A:** Validation trading_cost, minimized. Rate is computed per true class, so abandoning the rare ENTRY_SUB gives no discount, and a wrong-direction trade costs far more than a wrong-size one. A test proves the blind-but-accurate model scores worse.

`trainer/objective.py:78-80 (TITLE/SERIES/SIGN=min), trainer/objective.py:210-226 (trading_cost); wired at trainer/hpo.py:249-251`

### Hold-out validation for tuning — train | embargo | VAL | embargo | TEST
The search is scored on a separate validation slice, never on the test slice. The test slice is opened once, on the winner, at the very end. So we never tune on the number we report to the manager. There are two time-gaps (embargoes) so a val/test row can't share its rolling window with training rows.

**Why this, not the alternative:** If we picked the best TEST score over N trials, the test set enters the training loop — we'd be fitting the SETTINGS on it. The reported score would be the lowest of N noisy draws, biased low by ~2-3 cost points (the winner's curse). Validation absorbs that bias; test stays honest. Two embargoes because the 20-session feature lookback would otherwise make a val row a near-duplicate of training.

**Q (manager):** If you run dozens of trials and keep the best, isn't the reported score cherry-picked?  
**A:** The search only ever sees validation. Test is untouched during tuning and opened once on the winner, so that test number is the honest one I quote. The winner's val score is biased low by construction — that's the winner's curse, and it's why test exists.

`trainer/objective.py:84-197 (three_way_split); objective is val-only trainer/objective.py:78-80; leakage test tests/test_hpo_contract.py:174-188`

### Parameter search space — discrete, uniform, and integer ranges from one YAML
Every knob's allowed values live in one file, configs/hyperparams.yaml. A list means try exactly these values (DiscreteParameterRange). {min,max,step} means a plain range. Integer knobs use an integer range so no trial ever samples 7.4 for tree depth.

**Why this, not the alternative:** One YAML is the single source of truth. The old design had defaults in train.py and the space hardcoded separately in hpo.py; they drifted, and hpo.py searched knobs argparse had never heard of, so ClearML silently trained the default model. Integer ranges specifically stop argparse crashing when it parses a float depth back as int.

**Q (manager):** Where are the search ranges defined, and how do you stop them drifting from the training defaults?  
**A:** All in configs/hyperparams.yaml — the `default:` block feeds training, the `search:` block feeds HPO, side by side per model, so they can't disagree. Integer knobs use integer ranges so no trial sends a fractional tree depth that would crash the trial.

`trainer/hyperparams.py:169-225 (search_space builder); spaces configs/hyperparams.yaml:41-46 (rf), :66-75 (xgb), :97-107 (catboost)`

### Log-uniform (log-scale) sampling for learning_rate
learning_rate is searched on a log scale, not a straight line. That gives 0.01 and 0.1 equal attention. On a straight scale almost every random draw would land near the top and small learning rates would rarely get tried.

**Why this, not the alternative:** learning_rate spans orders of magnitude, so uniform sampling over-samples the big end. We store the real values (0.01..0.2) in YAML and take log10 in ONE place, because ClearML's LogUniformParameterRange actually wants EXPONENTS, not values — passing 0.01/0.2 raw would sample learning rates around 1.0-1.5 and train rubbish without erroring.

**Q (manager):** Why is learning rate searched differently from the other knobs?  
**A:** It ranges over orders of magnitude, so we sample it in log space to give small and large rates equal footing. We keep the real numbers in YAML and convert to exponents once in code, because the library takes exponents — getting that wrong silently trains garbage for hours.

`trainer/hyperparams.py:215-218 (log10 conversion); configs/hyperparams.yaml:68 (xgb log: true), :104 (catboost); trap test tests/test_hpo_contract.py:224-244`

### Result caching keyed by dataset content hash (parquet_sha256) — memoization
Each promoted tune records the SHA-256 of the exact parquet it was tuned on. When you publish a new version with --tune, if the dataset's SHA matches what a model was last tuned on, HPO is skipped and the old winner stands. If the features changed, the SHA is different and it re-tunes.

**Why this, not the alternative:** HPO is dozens of agent-hours per model. Re-running it when the data hasn't changed is pure waste. Making the DATA itself the cache key means nobody has to remember whether a re-tune is needed — a single changed byte in the features flips the SHA and forces a fresh search. --re-hpo overrides it.

**Q (manager):** How do you avoid re-running an expensive search on every single publish?  
**A:** The tuned file stores the parquet's SHA-256. Publish compares it to the current dataset's SHA: same data, skip HPO; changed data, re-tune automatically. It's content-addressed, so the data decides, not my memory.

`trainer/hyperparams.py:63-73 (tuned_sha); cache gate core/publish_version.py:369-380 (run_tune); SHA recorded trainer/apply_hpo.py:116`

### Pinning constants through HPO (dataset id/version + fixed seed 42)
Dataset id, dataset version, and seed=42 are forced into every trial as one-value ranges. So all trials train on identical rows with identical randomness, and the ONLY thing that changes between trials is the tuned knobs.

**Why this, not the alternative:** If data floated, we'd be comparing settings AND data at once and the winner would mean nothing. If the seed floated, the search would find the LUCKY seed that suits the validation slice and we'd ship a coin flip, not a better model. A one-value DiscreteParameterRange is the clean way to hold a constant across the whole search.

**Q (manager):** How do you know the winning trial won on its settings, not on easier data or a lucky random seed?  
**A:** Every trial is pinned to the same dataset id/version and seed 42 via one-value ranges, so hyperparameters are the only thing that varies. A test fails if the seed ever leaks into the search space.

`trainer/hpo.py:222-228 (pin dataset + seed=42); seed-exclusion test tests/test_hpo_contract.py:428-437`

### Trial budget and per-job time cap
The search runs a fixed number of trials — 30 by default when you run hpo.py directly, 15 when publish --tune drives it. Any single trial that runs longer than 90 minutes is killed. There is also an optional whole-search time limit.

**Why this, not the alternative:** Random search has no natural stopping point, so we bound it by total trials. A runaway forest (max_depth unlimited, min_samples_leaf=2) can sit for hours, so each job is time-capped. This is how spend on the paid agents is controlled, and every --tune run warns before it starts that it's real agent time.

**Q (manager):** What stops this running forever or blowing up the cloud bill?  
**A:** Trials are capped — 30 direct, 15 via publish — each trial is killed after 90 minutes, and there's an optional overall time limit. Before any trial runs, --tune prints a warning that it's up to N training jobs per model of real agent time.

`trainer/hpo.py:180 (trials 30), core/publish_version.py:610 (15 via --tune); caps trainer/hpo.py:263-264 (total_max_jobs, time_limit_per_job), trainer/hpo.py:187 (job_minutes 90)`

### Parallel trials capped to the number of agents
Trials run in parallel, but no more at once than you have agents. The controller loop runs on the laptop and just polls ClearML every minute; it does not itself occupy an agent.

**Why this, not the alternative:** Setting concurrency higher than the agent count only queues jobs — no faster. Running the controller locally with start() (not start_locally(), which runs trials in-process with no parallelism, and not --remote, which puts the controller ON an agent and deadlocks with a single agent) keeps every agent free to train.

**Q (manager):** How is the search parallelized across your machines?  
**A:** Concurrent trials are set to the number of clearml-agents; extra concurrency just queues. The controller runs locally and only polls, so it doesn't eat an agent — every agent stays free to train. Parallelism equals agent count.

`trainer/hpo.py:182-184 (--concurrent = agents), trainer/hpo.py:254 (max_number_of_concurrent_tasks), trainer/hpo.py:286-296 (start() runs controller locally)`

### HPO preflight — fail-fast on the two silent no-op failures
Before ANY trial runs, we check that the base task really has every parameter the search will override (each named 'Args/...'), that a validation set exists, and that a dataset is pinned. If any check fails, it stops immediately with a clear error instead of running.

**Why this, not the alternative:** ClearML only logs a warning if an override name is wrong or the objective metric is missing — then it trains the DEFAULT model, reports the default score, and 30 green trials change nothing. A warning in an agent log is a warning nobody reads. The preflight turns that silent warning into a hard stop before spending agent-hours. This is the same silent-prefix bug family that already cost the project a day.

**Q (manager):** What if a search knob is misnamed — does the whole tuning quietly do nothing?  
**A:** That's the exact failure we guard against. Every knob is named Args/<name> to match how ClearML files argparse params, and preflight refuses to start if any name is missing, if there's no validation set, or if no dataset is pinned. Tests pin the Args/ prefix and the objective string.

`trainer/hpo.py:124-170 (preflight); tests tests/test_hpo_contract.py:59-116`

### Manual winner promotion (apply_hpo) with a winner's-curse / range-edge guard
The search writes a best_params file but does NOT auto-apply it. A person runs apply_hpo.py to promote it into configs/tuned/<model>.json. It warns if the winning value sits at the very edge of its search range, or if the val score is far below the test score, and refuses to promote a range-edge winner unless you pass --force.

**Why this, not the alternative:** The winner is the lowest of N noisy draws (winner's curse). A value pinned to a range edge usually means the range was the binding limit, not the data — promoting it bakes in an artefact of the range you happened to pick. Auto-applying would ship those blind. The tuned file also records which run and which data SHA the numbers came from, so the promotion is on the record.

**Q (manager):** Do you just take whatever the search calls best and ship it?  
**A:** No — promotion is a deliberate manual step. It flags a winner that hit a range edge (widen and re-search instead) and flags a val score much lower than test (over-fitted search), and it records the source run and dataset SHA in the tuned file. The machine proposes, a human promotes.

`trainer/apply_hpo.py:80-106 (range-edge + val<test warnings, refuse without --force), trainer/apply_hpo.py:108-120 (tuned file records run + SHA)`


---

## 10. Feature selection

### Manual expert ballot (human / judgment-based feature selection)
A person picks the features, not code. `--new` prints a menu of every registered feature into selection_sheet.yaml with none ticked. The expert opens it and puts an `x` after the ones he wants (he types no feature names). `--from-sheet` reads the ticks and freezes exactly those into an immutable recipe. Nothing in the file scores, ranks, or auto-drops a feature.

**Why this, not the alternative:** The obvious alternative is automatic selection -- RFE or 'keep the top-N by importance/correlation'. It was rejected on purpose. In this data the highest-correlation columns are leakage, not alpha: registry.yaml's banned list shows columns at 0.30-0.37 correlation with the FUTURE return when real 1-minute alpha is 0.02-0.05. An importance ranker would pick exactly those poisoned columns. A human who knows the market (and knows which columns look into the future) is the safer filter.

**Q (manager):** So a machine never decides which features go into a dataset?  
**A:** Correct. A domain expert ticks a ballot. The code's only jobs are to check every ticked name is a real registered feature (freeze, lines 124-126) and that at least one is ticked (line 128). It never invents, ranks, or removes a feature on its own. The ballot even accepts `name:x` with no space -- it is parsed by hand (lines 236-245), not YAML, so a human filling a form does not have to know YAML.

`/home/megaserve/Desktop/Gourav/final_pipeline/core/make_version.py:208-247`

### Group / block selection (--group)
The feature team files each feature into a folder (its 'group', e.g. Bucket_Features, Bucket_Raw_Features). `--group NAME` grabs every feature in that group in one shot, no ticking. Comma-separate names to combine folders -- it takes their union. `--groups` first lists the buckets and how many features each holds, so you pick the exact name.

**Why this, not the alternative:** Ticking a whole bucket feature-by-feature is slow and easy to misspell. When the expert's unit of thought is 'use the raw-computed bucket', one word is safer than 30 ticks. It is still a human choice -- the human chooses the bucket -- just at a coarser grain. That is why each group build is recorded as a fresh MAJOR version (a new whole-menu decision), not a variation.

**Q (manager):** Where do the group names come from, and what if I name one that doesn't exist?  
**A:** From bridge/register.py -- it records each feature's sub-folder as its `group`. mode_group reads that field (m.get('group'), lines 294 and 301) and rejects any unknown group up front, printing the valid list (lines 296-300). v4 and v5 were built this way -- their recipes say selected_by: group:V4 and group:V5.

`/home/megaserve/Desktop/Gourav/final_pipeline/core/make_version.py:284-305`

### Explicit named-set selection (--feature)
Pick exact features by name, no folder and no ballot. `--feature combo_bucket_bucketraw` builds a version from just that one named parquet; comma-separate to put several into one version. This is the path for when each file the feature team hands over already IS a whole feature set -- one delivered parquet becomes one dataset version.

**Why this, not the alternative:** Forcing a single pre-combined parquet through the tick-a-ballot flow is pointless ceremony -- there is nothing to choose. Naming it directly maps one file to one version with no ambiguity. It is still fully manual; the human states the exact set.

**Q (manager):** What stops a typo here from silently building the wrong dataset?  
**A:** mode_feature checks every requested name against the registry before doing anything and lists what is registered if one is wrong (lines 318-321). Even if that were skipped, every mode ends in the same freeze(), which raises on any unknown name (lines 124-126). A misspelled feature stops the build loudly; it never silently drops.

`/home/megaserve/Desktop/Gourav/final_pipeline/core/make_version.py:308-323`

### Leave-one-out feature ablation (one-factor-at-a-time variations)
To learn whether a feature earns its place, you take an existing version and change exactly one thing: `--from v2 --drop stress_signal` makes v2.1, which is v2 minus that one feature. `--from-plan` runs many such single-change variations from one file in one command (an ablation sweep). The version number carries the meaning: v2 is a fresh pick (major), v2.1 is v2 with one change (minor).

**Why this, not the alternative:** This is the honest replacement for a model's automatic feature-importance chart. Importance charts are confounded -- correlated features split the credit -- and cannot be trusted when leakage is in play. Dropping one feature and re-measuring the actual score is a clean controlled test: only one thing moved, so a score change is attributable to that feature. The tool even refuses a no-op change that would produce a version identical to its parent (lines 265-267).

**Q (manager):** How do you guarantee two versions are a fair comparison?  
**A:** By the numbering rule: same MAJOR means only the one dropped/added feature differs, so it is fair; different MAJOR means too much changed, so don't compare (the rule is spelled out in the comment at lines 44-66). A variation also inherits its parent's frozen clocks rather than re-reading today's registry (lines 136-138 and 177), so bar alignment can't quietly drift between the two versions and contaminate the result.

`/home/megaserve/Desktop/Gourav/final_pipeline/core/make_version.py:250-275, 340-394`


---

## 11. Data & experiment versioning

### Recipe-not-copy (immutable frozen recipe / dataset-as-code)
We do not save a fresh copy of the data for every dataset version. Instead we save a small text file that says which features to use, which labels file, and each feature's bar clock. That file is the version. To rebuild the exact data, you re-run the build from the recipe plus the raw parquet. The recipe is never edited after it is written -- if you want a change you make a new version.

**Why this, not the alternative:** Copying the full 200 MB table for every experiment would explode storage (we expect crore-scale experiments). A recipe is a few hundred bytes and still reproduces the exact table, because the raw data and code are versioned too. Editing in place was rejected because the manifest certifies sha256(recipe), so any edit would break its own proof on the next read.

**Q (manager):** If you never copy the data, how do you know two people rebuilt the identical table?  
**A:** The recipe is hashed (recipe_sha256) and the built parquet is hashed (parquet_sha256). Same recipe plus same raw bytes gives the same parquet hash every time. If any of the three inputs differ, a hash differs and publish stops.

`core/make_version.py:119 (freeze), :180-183 (immutable write); docstring :16-18`

### Semantic major.minor versioning for single-variable ablation (PEP440-aligned)
Versions are numbered v1, v2 for a fresh human feature pick, and v2.1, v2.2 for a small change on top of v2 (one feature dropped or added). The rule: only compare versions with the SAME major number, because then exactly one thing changed and the score difference means something. Comparing v2 to v7 teaches nothing -- too much changed at once.

**Why this, not the alternative:** This makes every comparison a controlled experiment instead of a guess. It also maps 1-to-1 onto ClearML's version string (v2.1 becomes '2.1'), which ClearML sorts numerically (PEP440), so '2.10' correctly sorts after '2.9' and there is no translation code to get wrong. A bare 'v3' would not sort against 'v10'.

**Q (manager):** Why not just number them v1 v2 v3 and diff any two?  
**A:** Because a diff only proves cause if one variable moved. Same major guarantees one feature changed; different major means the selection changed wholesale and the result is uninterpretable. Deriving from v2.1 still becomes v2.3, flat, not v2.1.1 -- it is still a variation on v2, and the parent is recorded for full lineage.

`core/make_version.py:94-108 (next_major/next_minor), :44-66 (rationale); core/publish_version.py:74-85 (semver)`

### Content-addressed hashing (SHA-256 of bytes, not file timestamps)
Every file is identified by a SHA-256 hash of its actual contents, not by its modified-time. For notebooks we hash only the code cells, ignoring saved outputs and run counts, so a re-run with the same code still hashes the same.

**Why this, not the alternative:** Modified-time lies. Copying a feature file off Windows bumps its timestamp although the bytes are identical (a needless rebuild), and restoring an old file can make it look older than a stale cache (a silently stale feature reaching the live model). Hashing the bytes answers the honest question: did the content actually change?

**Q (manager):** Why is a timestamp not good enough to detect a changed feature?  
**A:** Timestamps change on copy and restore even when the data is identical, and can go backwards. That caused both false rebuilds and, worse, stale features slipping through. A content hash only changes when the bytes change.

`hashes.py:17-23 (sha256_file), :34-52 (notebook code-only hash)`

### DVC content-addressed data layer with ClearML external-file pointer (single copy)
The dataset bytes are pushed to our own GCS bucket by DVC, which stores files by their content hash. ClearML does NOT get a copy of the data -- it is registered with an external-file link that points at the DVC copy in the bucket. So there is exactly one copy of the data, and it lives in our bucket.

**Why this, not the alternative:** It fixes ClearML's weak spot (data de-dup) for free and keeps the data on our own GCP -- the SaaS ClearML server only ever holds the pointer and metadata, never the bytes. DVC is content-addressed and needs no server. In local mode the bytes go to a self-hosted ClearML fileserver instead, still never leaving our machines.

**Q (manager):** Does our raw market data leave our infrastructure when you use ClearML SaaS?  
**A:** No. add_external_files registers only a gs:// pointer; the bytes were already pushed to our bucket by dvc push. app.clear.ml holds metadata and the pointer, nothing else.

`core/publish_version.py:445-468 (dvc add/push + get_url), :548 (add_external_files pointer)`

### Manifest certificate with a validate-on-publish gate
The feature team hands over two files per version: the parquet and a manifest.json. The manifest is a certificate -- it lists the schema, row count, feature count, three checksums (recipe, parquet, labels), per-feature provenance and clocks, and the class distribution. Before publishing, the core re-checks every claim: recipe hash matches the recipe on disk, parquet hash matches the file, and rows/columns/feature-count are read straight from the parquet footer and compared. Any mismatch and publish stops dead.

**Why this, not the alternative:** A certificate that only records claims without checking them is worthless -- an earlier version carried rows and column names but never verified them against the parquet. Now the core trusts nothing it did not re-verify, so a bad or swapped dataset can never reach the model. The footer check is cheap: pyarrow reads only the metadata, not the whole 200 MB file.

**Q (manager):** What stops someone publishing a parquet that was built from a different feature selection?  
**A:** The recipe_sha256 in the manifest must equal the hash of the recipe on disk. If the parquet was built from a different recipe, that hash won't match and validate_manifest raises RECIPE MISMATCH before anything is published.

`core/contract.py:57-132 (validate_manifest), :24-29 (required keys)`

### No-lookahead sworn statement (no_peek: bar_close)
The manifest carries a small block that swears the no-lookahead alignment rule was actually applied, and names the rule ('bar_close'). Publish refuses if the block says applied=false, or names any rule other than bar_close.

**Why this, not the alternative:** Lookahead is the bug that backtests beautifully and loses money live -- a 5-minute bar must only be used after it closes. It is not enough to claim lookahead is absent; the certificate must positively swear the correct rule ran, and the core only trusts the one rule it has proven correct. Anything else is refused rather than assumed safe.

**Q (manager):** How do you guarantee no future information leaked into a training row?  
**A:** The build applies the bar_close rule and stamps no_peek.applied=true, rule='bar_close' in the manifest. publish_version refuses any dataset where applied is false or the rule is not exactly 'bar_close', so an un-aligned dataset cannot be published.

`core/contract.py:101-107; manifest no_peek block (applied/rule/tolerance_bars)`

### ClearML dataset versioning with mandatory finalize()+publish() ordering and reuse-repair
Registering a version in ClearML follows a strict order: add the file link, upload, finalize, then publish. Finalize marks it 'completed'; publish marks it 'published'. The auto-train trigger only listens for 'published'. If a re-run finds a version stuck at 'completed' (finalized but not published), the code publishes it; if it finds a half-built version, it refuses and tells you to delete it.

**Why this, not the alternative:** Stopping at finalize() is a silent killer -- the dataset looks healthy in the UI, the trigger waits for 'published' forever, and nothing trains, with no error. That cost a full day once. The order also matters because add_external_files marks the dataset dirty and finalize refuses a dirty dataset; upload clears the flag. The reuse path checks real status on the backing task because a ClearML Dataset object has no reliable .status attribute.

**Q (manager):** You said a dataset was created but the model never trained -- what was wrong and how is it prevented now?  
**A:** It was left at finalize() (status 'completed'); the trigger only fires on 'published'. Now we always call publish() after finalize(), a unit test pins that contract against library upgrades, and the reuse branch auto-publishes a stuck 'completed' version.

`core/publish_version.py:558-560 (finalize+publish), :515-538 (reuse-path status repair)`

### Lock-back receipt (provenance lock file)
After a successful publish, a separate file dataset_vN.lock.yaml is written. It records the ClearML dataset id, all three checksums (parquet, recipe, labels), the git commit, and the bucket URL. This is the record that answers 'which exact data, code and labels did model X train on?'.

**Why this, not the alternative:** It lives in its OWN file, never inside the recipe -- because the manifest certifies the hash of the recipe, and writing into the recipe would break that hash on the next read. Keeping the receipt beside the recipe preserves both the immutable recipe and a permanent provenance trail.

**Q (manager):** Six months from now, how do you prove exactly what data a deployed model was trained on?  
**A:** Open its version's lock file. It pins the ClearML dataset id, the parquet/recipe/labels sha256s, and the git commit. Those four together reproduce the exact training input; nothing about it can drift because the recipe is immutable.

`core/contract.py:165-192 (write_lock); core/publish_version.py:564-567`

### Immutable-version byte guard (one version = one set of bytes)
If a version was already published once and you rebuild its parquet so the bytes change, republishing is refused. The message tells you to make a new version instead.

**Why this, not the alternative:** The ClearML dataset for that version still points at the OLD bytes. Silently reusing its id would train the models on data that no longer matches the manifest, and the lock would swear everything matched -- a lie waiting to happen. A version number must mean one fixed set of bytes forever.

**Q (manager):** What if I quietly rebuild v3 with a tweak and republish under the same number?  
**A:** It's blocked. publish compares the parquet's current sha256 against the sha256 stored in v3's lock; if they differ it stops and tells you to run make_version --from v3 to get a new number. Same version can never point at two different datasets.

`core/publish_version.py:417-434 (locked-sha vs current-sha guard)`


---

## 12. Explainability (SHAP)

### SHAP (Shapley values) via TreeExplainer
SHAP gives every feature a number for one single prediction. The number says how hard that feature pushed the model toward or away from an answer. Add up all the feature numbers plus a base value and you get the exact prediction back. For tree models it uses TreeExplainer. Random forest and xgboost go through shap.TreeExplainer; catboost has its own built-in SHAP, which is faster and better tested on catboost.

**Why this, not the alternative:** The model's own built-in feature_importance gives one global score per feature and can never tell you why one specific losing trade happened. TreeExplainer is exact for tree models, so there is no approximation error, unlike the general-purpose KernelExplainer which only estimates by sampling. That is why it was chosen.

**Q (manager):** Is SHAP guessing which features matter, or is it exact?  
**A:** For tree models TreeExplainer computes exact Shapley values, not an estimate. The only place we sample is choosing which rows to explain (SHAP is slow), never the attribution math itself.

`trainer/shap_logic.py:79 (shap.TreeExplainer), trainer/shap_logic.py:72 (catboost ShapValues), called at trainer/shap_explain.py:179`

### Cost-weighted error ranking (importance = error_rate x severity)
For every kind of mistake (true class was A, model said B) we count how often it happens as a fraction of all the real A rows, then multiply by how much that confusion costs. The cost comes from a config file. A high final number is where real money is bleeding.

**Why this, not the alternative:** A plain confusion matrix or an accuracy score treats every mistake as equal. In trading they are not: buying big when you should sell big can wipe the account, while calling a small trade wrong barely costs anything. Multiplying rate by cost points SHAP at expected money lost, not raw error count.

**Q (manager):** Where do the severity numbers come from -- did you just invent them?  
**A:** From configs/severity_7class.json, tiered on trading logic: a full reversal at max size is 100, a mild under-size is 3. The desk can edit the file; the code just obeys it (trainer/shap_explain.py:148).

`trainer/shap_logic.py:90-121 (rank_mistakes, importance at :114), called at trainer/shap_explain.py:149`

### Tail-risk / worst-case screen (high severity, any rate)
A second, separate list. It shows every high-cost mistake that happened even once, no matter how rare. It exists because the first ranking is an average, and an average can hide a rare disaster behind a pile of small common ones.

**Why this, not the alternative:** rate x severity measures expected total damage, which is right on average, but it can bury a catastrophe: 40 nuisance trades outscore 2 full reversals even though a full reversal at max size is what ends a trading book. So we keep two views -- 'where the money goes' and 'what could kill us' -- instead of trusting one number.

**Q (manager):** If your main ranking is already rate times cost, why do you need a second list?  
**A:** Because rate x severity is an average and averages hide tail risk. Worked example in the code: 2 reversals score 2.0, 40 nuisance trades score 3.2, so the reversal ranks lower even though it can end the book. This list surfaces anything with severity >= 50 regardless of how rare.

`trainer/shap_logic.py:124-144 (worst_case_mistakes), called at trainer/shap_explain.py:161`

### Mean-absolute-SHAP feature importance, normalized to shares (per-model SHAP space)
For each feature take the average size of its SHAP value across the rows, ignoring the plus/minus sign, then divide by the total so it becomes a percent. A big percent means that feature is carrying the model's decisions. We compare percents, not the raw SHAP sizes.

**Why this, not the alternative:** The three models do not speak the same units: random forest SHAP is in probability (0 to 1), xgboost and catboost are in log-odds (can be -3.86). Comparing raw numbers would just tell you which model sits on a bigger scale, not which feature matters. Converting each model to a share of its own total makes them comparable. The SHAP_SPACE map records which unit each model is in.

**Q (manager):** Can you compare feature importance across the three models directly?  
**A:** Not the raw SHAP numbers -- random forest is in probability, xgboost and catboost in log-odds. We normalize each model to its own total and compare shares (percentages). The unit of each model is tracked in SHAP_SPACE so we never mix them.

`trainer/shap_logic.py:147-161 (feature_shares), SHAP_SPACE at trainer/shap_logic.py:42-46, units note at trainer/shap_logic.py:17-27`

### Bootstrap stability of feature shares (stability selection)
Run the share calculation 5 times, each time on a different random 70% slice of the sampled rows. Then report the average share, how much it wobbled between runs, and how many of the 5 runs put the feature in the top 5. Verdict: 'solid' means top-5 in all 5 runs, 'noise' means it got lucky once.

**Why this, not the alternative:** One SHAP ranking always looks confident, but it was built on a sample, so some of the order is luck. Without a wobble number you can't tell a real driver from a lucky one, and you might waste time re-engineering a feature that only ranked 4th by chance. Measured on synthetic data: the top feature is rock solid across runs, ranks 4-5 move ~5%.

**Q (manager):** How do you know the top feature isn't just an artifact of which rows you happened to sample?  
**A:** We recompute it on 5 overlapping 70% subsamples. A feature is only called 'solid' if it lands in the top 5 in every one of the 5 runs, and we print the run-to-run standard deviation next to every share so you can see which ranks are shaky.

`trainer/shap_logic.py:164-203 (stable_feature_shares, n_boot=5, frac=0.7), called at trainer/shap_explain.py:187`

### SHAP waterfall plot (single-prediction attribution)
A picture for one actual wrong trade. It starts at the base value and stacks each feature's push, up or down, until it reaches the model's final answer. You see exactly which features shoved that one prediction into the wrong class.

**Why this, not the alternative:** Aggregate importance tells you which features matter overall, but a waterfall tells the story of one real mistake, which is what actually convinces a human the model is wrong for a nameable reason. We only draw it for the exact true->predicted pair we are chasing, and save it as a PNG because ClearML silently drops parts when it converts matplotlib figures to plotly.

**Q (manager):** Are those waterfall charts real mistakes or made-up examples?  
**A:** Real test rows the model actually got wrong, filtered to exactly the worst true-to-predicted pair (trainer/shap_explain.py:219). Base value plus the feature pushes equals the model's real output for that row.

`trainer/shap_explain.py:221-223 (shap.Explanation + shap.plots.waterfall), filtered to the A<->B pair at trainer/shap_explain.py:219`

### Stratified sampling of the worst pair (for SHAP speed)
We do not run SHAP on all ~500k test rows. We take a few hundred rows from each of the two classes involved in the worst mistake and explain only those. Same pattern, tiny fraction of the time.

**Why this, not the alternative:** SHAP on 500k rows times 7 classes would take hours and tell us nothing extra. The mistake we're explaining only involves two classes, so we sample just those two, evenly, a few hundred each. A fixed seed makes the run reproducible so results don't change between runs.

**Q (manager):** If you only explain a sample, aren't you cherry-picking the rows that suit your story?  
**A:** No. The sample is drawn with a fixed seed (42) evenly from the two classes in the worst pair -- not hand-picked. And the SHAP attribution itself stays exact; we only sample which rows to look at, not the math.

`trainer/shap_logic.py:213-227 (sample_for_shap, seed=42), called at trainer/shap_explain.py:173`
