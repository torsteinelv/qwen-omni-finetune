# Prosjektplan: Finjustering av Qwen-Omni for norsk tale

Målet med prosjektet er å gi **Qwen2.5-Omni-3B** evnen til å generere og forstå naturlig norsk tale ved å trene modellens "Thinker"-del til å produsere riktige lyd-tokens (Mimi).

## 📋 Gjennomføringsplan

### Fase 1: Datainnsamling og klargjøring
* **Hente datasett**: Bruke åpne kilder som Stortingets talekorpus (NPSC) og Google FLEURS for å sikre bred dekning av norsk språk.
* **Lydbehandling**: Resampling av all lyd til 24kHz for å matche kravene til Mimi-encoderen.
* **Strukturering**: Generere metadatafiler (JSONL) som kobler tekstlig innhold mot lydfiler.

### Fase 2: Audio-tokenisering (Talker-forberedelse)
* **Konvertering**: Bruke Mimi-modellen til å oversette rå lydfiler til diskrete lyd-tokens.
* **Datasett for "Talker"**: Opprette et spesialisert treningsdatasett der modellen lærer å forutsi disse tokensene basert på norsk tekst-input.

### Fase 3: Trening med LoRA og kvantisering
* **Effektivisering**: Implementere 8-bit eller 4-bit kvantisering for å redusere minnebruk under trening.
* **Adapter-trening**: Bruke LoRA (Low-Rank Adaptation) for å kun trene de mest kritiske lagene i "Thinker"-modellen.
* **Modell-wrapping**: Bruke en spesialtilpasset wrapper for å håndtere multimodale lag uten krasj ved lagring.

### Fase 4: Evaluering og testing
* **Inference-testing**: Generere tale fra tekst ved bruk av den ferdige adapteren for å sjekke kvalitet og naturlighet.
* **Sammenligning**: Teste mot den originale basemodellen for å dokumentere forbedringen i norsk uttale.

---

## 🚦 Statusrapport

### ✅ Gjennomført
* **Infrastruktur**: Docker-miljøet er satt opp med alle nødvendige biblioteker som `transformers`, `peft` og `bitsandbytes`.
* **Automatisering**: Et entrypoint-script er ferdigstilt som kjører hele rørledningen fra data til trening.
* **Data-scripts**: Scripts for behandling av både NPSC og FLEURS er operative.
* **Treningslogikk**: Implementert "Talker-only" trening med LoRA og minnehåndtering (Memory Fix).
* **CI/CD**: GitHub Actions er konfigurert for automatisk bygging av Docker-imager.

### ⏳ Pågående / Neste steg
* **Fullskala trening**: Kjøre trening på det utvidede NPSC-datasettet (mål: 15 000 klipp).
* **Optimalisering**: Finjustere hyperparametre for bedre talekvalitet i norsk kontekst.
* **Validering**: Gjennomføre omfattende tester av generert lyd via `test_speak.py`.
