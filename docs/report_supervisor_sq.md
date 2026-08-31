# Raport i progresit — kodi dhe rezultatet

**Projekti:** Përzgjedhje e nëngrafeve e bazuar në qëllim për përgjigje pyetjeve
mbi grafe klinike të dijes
**Autori:** Serxhio Dosku, MSc, Universiteti i Bazelit
**Data:** 31 gusht 2026 (java 7 nga 24)

---

## 1. Përmbledhje

Bërthama e sistemit të kërkimit është ndërtuar, testuar dhe ekzekutuar kundrejt
grafit real STCS. **274 teste automatike kalojnë me sukses.** Të gjitha
rezultatet eksperimentale deri tani vijnë nga një graf sintetik që riprodukton
saktësisht strukturën e matur të një pacienti real, prandaj ato përshkruajnë se
si sillen metodat mbi këtë klasë grafi, dhe jo cilësinë e kërkimit klinik.

Një çështje kërkon vëmendjen tuaj: **shtegu i grafit i SnapQuery nuk funksionon
në serverin CHIL** (seksioni 5).

---

## 2. Çfarë është ndërtuar

| Komponenti | Statusi |
|---|---|
| Përzgjedhja e nëngrafit me PCST (G-Retriever) | Verifikuar identik bajt për bajt me implementimin e publikuar |
| Katër strategji bazë | Top-*k* tripleta (KAPING), top-*k* nyje + fqinjë, zgjerim BFS, shtigjet më të shkurtra |
| Ngarkuesi Neo4j për grafin SPHN | Punon vetëm për lexim kundrejt grafit real STCS |
| Metrika e kthimit të nyjeve-përgjigje | Pyetjet agregate regjistrohen si të papërcaktuara, nuk vlerësohen zero |
| Ndërlidhja e terminologjive | Zbërthen kodet ICD/LOINC/ATC në përshkrime, në pesë nivele të raportuara |
| Generim kandidatësh në dy faza | Bën përzgjedhjen të realizueshme në shkallë pacienti |
| Replika me formë të matur | Riprodukton saktësisht pacientin #0, kështu puna vazhdon pa server |
| Klienti i SnapQuery | I plotë dhe i testuar; nuk mund të ekzekutohet ende (seksioni 5) |
| Formati dhe validimi i setit të artë | Gati për futjen e pyetjeve |
| Cikli i vlerësimit mbi kohortën | I riparueshëm, izolon dështimet për pacient |
| Figurat | Katër, të gjeneruara nga skedarët e rezultateve |

Çdo akses në bazën e të dhënave është vetëm për lexim. Asnjë e dhënë pacienti
nuk del nga BioMedIT: në tezë shfaqen vetëm emra skemash, emra vetish dhe numra
agregatë.

---

## 3. Testet

`pytest` — **274 teste, të gjitha kalojnë**, rreth 12 sekonda.

| Zona | Teste |
|---|---|
| Seti i artë dhe cikli mbi kohortën | 45 |
| PCST, përfshirë ekuivalencën me kodin e publikuar | 32 |
| Kërkuesit dhe metrika e vlerësimit | 24 |
| Generimi i kandidatëve | 19 |
| Zbërthimi i terminologjisë | 19 |
| Tipi i grafit dhe kodifikuesit | 18 |
| Klienti i SnapQuery | 17 |
| Besnikëria e replikës | 16 |
| Ngarkuesi Neo4j | 15 |
| Figurat | 13 |
| Strategjitë bazë | 10 |

Testet e PCST përfshijnë një kontroll ekuivalence: një kopje fjalë për fjalë e
rutinës së publikuar të G-Retriever ruhet si orakull testimi, dhe implementimi
i yni duhet të kthejë të njëjtat nyje, të njëjtat brinjë dhe të njëjtin
serializim, mbi grafe të përzgjedhura dhe të gjeneruara rastësisht. Çdo
diferencë është dështim testi.

---

## 4. Matjet dhe rezultatet

### Grafi (pyetje vetëm-lexim në CHIL, 19–20 gusht)

| Madhësia | Vlera |
|---|---|
| Nyje / brinjë | 21,537,305 / 72,218,890 |
| Pacientë | 1,197 |
| Një konstrukt laboratorik i reifikuar | 95.3% e të gjitha nyjeve |
| Një pacient pas përgatitjes | 15,810 nyje, 45,562 brinjë |
| Përshkrime të dallueshme për ato nyje | 614, rreth 26 nyje për përshkrim |
| Diagnoza pa përshkrim | 178 nga 222 (80%) |

### Rezultate eksperimentale (replika sintetike)

1. **Me parametrat e paracaktuar të publikuar, PCST kthen të gjithë grafin** —
   15,810 nga 15,810 nyje. Buxheti i çmimeve për brinjë ndahet mes të gjitha
   brinjëve me pikë të njëjta, dhe me pak tipa relacionesh të përdorura shumë,
   kostoja për brinjë bie në një vlerë të papërfillshme.

2. **Në madhësi të njëjtë nëngrafi, lidhshmëria nuk dha kontribut.** Në rreth
   njëmbëdhjetë nyje, PCST, shtigjet më të shkurtra, top-*k* nyje me fqinjë dhe
   KAPING dhanë saktësisht **0.164**. Ajo që kthehet vendoset nga sinjali i
   ngjashmërisë, jo nga strategjia strukturore.

3. **Kjo qëndron edhe mbi shumë pacientë.** Mbi dhjetë pacientë me 6,325 deri
   18,972 nyje, tri strategji përputhen në katër shifra dhjetore: mesatarja
   **0.1461**, devijimi standard **0.0081**.

4. **Zbërthimi i terminologjisë është vendimtar.** Diagnozat që mbajnë vetëm një
   kod kaluan nga **0.00 në 0.80** kthim, pasi kodet u zbërthyen në përshkrime.
   Përfitimi qe identik për çdo strategji kërkimi, gjë që e vendos këtë hap në
   përgatitjen e grafit dhe jo në kërkim.

5. **Një katalog në gjuhën e gabuar është më keq se pa katalog.** Aplikimi i një
   katalogu në anglisht mbi pyetje në gjermanisht i çoi diagnozat me emër nga
   **1.00 në 0.00**. Kjo është arsyeja pse duhet të dimë në çfarë gjuhe shkruajnë
   në të vërtetë përdoruesit.

6. **Generimi i kandidatëve nuk kushton.** Zgjerimi nga bërthamat prodhoi të
   njëjtin nëngraf me të njëjtin kthim, tri deri katër herë më shpejt. Por çdo
   generator hodhi tej çdo diagnozë vetëm-me-kod, prandaj zbërthimi i
   terminologjisë duhet të kryhet i pari.

---

## 5. SnapQuery në CHIL — kërkon vëmendjen tuaj

Shërbimin e matëm më 30 gusht. Është i arritshëm në portën 8002, nuk kërkon
autentikim nga serveri, dhe përgjigjet në 6 deri 17 sekonda për hap. Kontrata e
kërkesës është `POST /chat/` me `query` dhe një `session_id` të lëshuar nga
klienti, më pas `POST /chat/continue` vetëm me identifikuesin e sesionit.

**Shtegu i grafit nuk përfundon një pyetje.** `POST /chat/` kthen arsyetimin e
papërpunuar të modelit planifikues, me një thirrje mjeti **të paekzekutuar** në
fushën `answer`, dhe `POST /chat/continue` më pas kthen HTTP 500 pas rreth 30
sekondash.

Shkaku i mundshëm është një ndryshim modeli. Porta 8001 raporton
**`Qwen3.8-27B-FP8`**, ndërsa dokumenti i arkitekturës e përshkruan modelin
planifikues si Qwen2.5-14B-Instruct. Qwen3 lëshon blloqe arsyetimi `<think>`, të
cilat i shohim në daljen, dhe thirrjet e mjeteve u kthyen si
`<function=...><parameter=...>` dhe jo si JSON brenda `<tool_call>`. Një parser
i shkruar për Qwen2.5 nuk do ta lexonte këtë format, gjë që shpjegon njëkohësisht
thirrjen e paekzekutuar të mjetit dhe gabimin 500.

Identifikuesi i sesionit `bd28adf9-125f-4d58-b84c-51c1c8ebe3a1`, në rast që
ndihmojnë logjet.

Vlen të shënohet gjithashtu: ai model raporton `max_model_len` prej **8,192
tokenësh**, të ndarë mes skemës reale në prompt-in e sistemit dhe arsyetimit të
vetë modelit.

Klienti i yni është shkruar dhe testuar kundrejt fiksurave të regjistruara,
kështu që kur shërbimi funksionon, ndryshon vetëm një komponent. Deri atëherë
krahasimi nuk mund të kryhet.

---

## 6. Çfarë kërkojmë prej jush

1. **A pritet të funksionojë shtegu i grafit i SnapQuery në CHIL, dhe a ka
   ndryshuar modeli planifikues?** Kjo bllokon krahasimin kryesor të tezës.
2. **A ruhen pyetjet e përdoruesve, në çfarë forme, dhe kush mund t'i
   eksportojë?** Vlerësimi varet nga pyetje reale.
3. **Kush mund të validojë rreth 50 pyetje të setit të artë?** Formati dhe
   kontrollet janë gati; kjo kërkon kohën e një klinicisti ose menaxheri të
   dhënash.
4. **Cilat etiketa janë brenda fushës?** Ngarkuesi aktualisht injoron `Sample`
   (6.0M nyje), `DrugPrescription` (35,784) dhe demografikët. Kjo përjashton
   pyetjet për mostra, për të përshkruar kundrejt të administruar, dhe për
   vdekshmërinë.
5. **A është `BilledDiagnosis` etiketa e saktë e diagnozës**, apo duhen
   përdorur diagnozat e regjistruara klinikisht?
6. **A mund të publikohen numrat agregatë?** Teza raporton numra nyjesh dhe
   frekuenca etiketash, pa informacion në nivel pacienti. Do të donim
   konfirmimin tuaj të dokumentuar.
