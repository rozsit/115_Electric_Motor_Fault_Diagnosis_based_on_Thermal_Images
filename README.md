# 115_Electric_Motor_Fault_Diagnosis_based_on_Thermal_Images
Electric_Motor_Fault_Diagnosis_based_on_Thermal_Images_with_Machine_Learning

## Dataset: https://www.kaggle.com/datasets/python16/electric-motor-thermal-image-fault-diagnosis/data


📊Hőtérképekből gépi tanulással: villanymotor-hibák előrejelzése.

Az ipari prediktív karbantartás egyik “titkos fegyvere” a termográfia: a motorok külső hőeloszlása (hőtérképe) sokszor hamarabb elárulja a hibát, mint bármelyik zaj vagy teljesítménycsökkenés. Most egy olyan Colab-notebookot raktam össze, amely villanymotorok termálképeiből tanul és hiba / nincs hiba osztályozást végez kevés tanítóadat mellett is stabilan.

1️⃣Mit csinál a notebook?

📍Adatforrás: IR/termál képek villanymotorokról, train/ és validation/ mappastruktúrában (binary: fault vs no_fault)

📍Előfeldolgozás & augmentáció: egységesítés (224×224), ImageNet-kompatibilis normalizálás, majd célzott augmentáció (forgatás, eltolás, zoom, nyírás, tükrözés) a kevés minta ellen

📍Modell: ResNet50 backbone (ImageNet súlyok, include_top=False) + GlobalAveragePooling + Dropout(0.4) + L2-regularizált sigmoid kimenet (binary)

📍Tanítás kis adaton: fagyasztott backbone-nal bemelegítés (Adam, BCE + label smoothing 0.05, metrikák: Accuracy, AUC)

📍Finomhangolás: a felső ~50 ResNet-réteg “feloldása” (BatchNorm kivétel), alacsony LR, korai megállítás (EarlyStopping), ReduceLROnPlateau

📍Osztály-súlyozás a kiegyensúlyozottabb tanulásért

📍Értékelés: Precision-Recall görbéből F1-optimális döntési küszöb kiválasztása (nem csak 0.5!), konfúziós mátrix és jelentés

📍Extra: adat-szivárgás ellenőrzés (azonos fájlnevek, bitegyezés SHA256-tal) a train/val közt

2️⃣Inference / üzemeltetés: 1-képes predikció, a kimenet ráírva az eredeti hőtérképre, mentéssel

3️⃣Hőtérképek (termográfia) létrehozása és szerepük:

📌Létrehozás: IR kamerával rögzített radiometrikus felvételek → egységes hőskála/kolormap → zajszűrés, opcionális ROI (motorház) kivágás → 3-csatornás bemenet a CNN-nek

📌Miért hasznos? A lokális forrópontok és aszimmetriák gyakran megelőzik a meghibásodást:

csapágymelegedés → csapágy-/kenési problémák,

tekercshelyi hotspot → szigetelési/tekercselési gond,

egyenetlen hőeloszlás → hűtés/projektálási anomália, terhelés-aszimmetria.

 📌ezzel a tervezett leállás időzíthető, csökken a nem tervezett kiesés

4️⃣Stratégiák kevés tréningképhez

Transzfer tanulás (ImageNet-en előtanított ResNet50) → kevesebb adat mellett is jó generalizáció.

Erős, de realistán beállított augmentáció → “mesterséges” variancia a valóságon belül.

Regularizáció (Dropout + L2) és label smoothing → kevés mintán is robusztusabb döntési határ.

Osztálysúlyok + F1-alapú küszöb → az üzemi kockázatokhoz illesztett hiba/álhiba kompromisszum.

Korai megállás, LR-léptetés → túltanulás megelőzése, stabil konvergencia.

Szivárgás-kontroll → a validáció valóban valós teljesítményt mér.

Mit ad ez a karbantartásnak?

Korai jelzés: a hőtérkép-mintázatok változását a modell hamar észreveszi.

Skálázhatóság: a pipeline Colabban fut, könnyen konténerizálható és integrálható edge/gyári környezetbe.
