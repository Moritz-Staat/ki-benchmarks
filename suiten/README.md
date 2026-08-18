# Aufgabensammlungen

Ein Verzeichnis je Suite, darin ein Verzeichnis je Aufgabe:

    <suite>/<schluessel>/aufgabe.json     Titel, Auftrag, Schwierigkeit, Schritte
    <suite>/<schluessel>/start/           Ausgangszustand, wird in die Sandbox kopiert
    <suite>/<schluessel>/tests/           entscheidet ueber bestanden / nicht bestanden

**`tests/` liegt getrennt von `start/`, und das ist kein Ordnungsfimmel.** Der
Runner kopiert nur `start/` in die Sandbox; die Tests kommen erst dazu, wenn das
Modell fertig ist. Laegen sie von Anfang an daneben, koennte ein Modell die
Tests umschreiben statt die Aufgabe zu loesen - und wuerde dafuer auch noch mit
einem gruenen Ergebnis belohnt.

## Die Suiten

| Suite | Zweck |
|---|---|
| `rauchtest` | **Keine Messsammlung.** Zwei triviale Aufgaben, nur damit Runner, Sandbox und Auswertung nachweisbar durchlaufen. Fuer einen Vergleich zwischen Modellen ist sie viel zu klein und zu leicht. |
| `sandbox-probe` | Eine einzige Aufgabe, deren Testcode absichtlich aus der Sandbox ausbrechen will. Sie **muss** scheitern - das ist die Abnahmebedingung aus Prompt C, nicht ein Fehler. |

Die eigentliche Sammlung mit 20-30 gestaffelten Aufgaben fehlt noch. Sie soll
auf `..\..\NOTIZEN.md` aufbauen - auf Aufgaben, an denen die Modelle im echten
Gebrauch gescheitert sind. Eine ausgedachte Sammlung waere schneller gebaut und
deutlich weniger wert.
