"""Generators and validators for Italian structured identifiers.

All generated values pass the official checksum so that validating recognizers
(Presidio IT_FISCAL_CODE / IBAN_CODE, python-stdnum) treat them as genuine.
"""
from __future__ import annotations

import random
import re
from datetime import date

VOWELS = set("AEIOU")
MONTH_CODES = "ABCDEHLMPRST"  # Jan..Dec

_ODD = {  # codice fiscale check-character table for odd positions (1-based)
    '0': 1, '1': 0, '2': 5, '3': 7, '4': 9, '5': 13, '6': 15, '7': 17, '8': 19, '9': 21,
    'A': 1, 'B': 0, 'C': 5, 'D': 7, 'E': 9, 'F': 13, 'G': 15, 'H': 17, 'I': 19, 'J': 21,
    'K': 2, 'L': 4, 'M': 18, 'N': 20, 'O': 11, 'P': 3, 'Q': 6, 'R': 8, 'S': 12, 'T': 14,
    'U': 16, 'V': 10, 'W': 22, 'X': 25, 'Y': 24, 'Z': 23,
}
_EVEN = {c: i for i, c in enumerate("0123456789")} | {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def _consonants_vowels(s: str) -> tuple[str, str]:
    s = re.sub(r"[^A-Z]", "", s.upper().replace("À", "A").replace("È", "E").replace("É", "E")
               .replace("Ì", "I").replace("Ò", "O").replace("Ù", "U"))
    return "".join(c for c in s if c not in VOWELS), "".join(c for c in s if c in VOWELS)


def cf_surname_code(surname: str) -> str:
    c, v = _consonants_vowels(surname)
    return (c + v + "XXX")[:3]


def cf_name_code(name: str) -> str:
    c, v = _consonants_vowels(name)
    if len(c) >= 4:
        c = c[0] + c[2] + c[3]
    return (c + v + "XXX")[:3]


def cf_check_char(partial: str) -> str:
    total = 0
    for i, ch in enumerate(partial, start=1):
        total += _ODD[ch] if i % 2 == 1 else _EVEN[ch]
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[total % 26]


def codice_fiscale(surname: str, name: str, birth: date, female: bool, belfiore: str) -> str:
    day = birth.day + (40 if female else 0)
    partial = f"{cf_surname_code(surname)}{cf_name_code(name)}{birth.year % 100:02d}{MONTH_CODES[birth.month - 1]}{day:02d}{belfiore}"
    return partial + cf_check_char(partial)


def validate_cf(cf: str) -> bool:
    cf = cf.upper()
    return bool(re.fullmatch(r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]", cf)) and cf_check_char(cf[:15]) == cf[15]


# ----------------------------------------------------------------- IBAN

def _iban_checksum(country: str, bban: str) -> str:
    s = bban + country + "00"
    num = "".join(str(int(ch, 36)) for ch in s)
    return f"{98 - int(num) % 97:02d}"


def iban_it(rng: random.Random) -> str:
    cin = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    abi = rng.choice(["05428", "03069", "02008", "01030", "05034", "03268", "08327", "06230", "03111", "05387"])
    cab = f"{rng.randint(10000, 99999)}"
    acct = f"{rng.randint(0, 999999999999):012d}"
    bban = cin + abi + cab + acct
    return "IT" + _iban_checksum("IT", bban) + bban


def validate_iban(iban: str) -> bool:
    iban = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    s = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in s)) % 97 == 1


def format_iban(iban: str, rng: random.Random | None = None) -> str:
    """Return the IBAN either compact or grouped by 4 (both occur in real documents)."""
    if rng is not None and rng.random() < 0.5:
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
    return iban


# ----------------------------------------------------------------- Partita IVA

def piva_check_digit(first10: str) -> str:
    total = 0
    for i, ch in enumerate(first10):
        d = int(ch)
        if i % 2 == 0:
            total += d
        else:
            d2 = d * 2
            total += d2 - 9 if d2 > 9 else d2
    return str((10 - total % 10) % 10)


def partita_iva(rng: random.Random) -> str:
    first10 = f"{rng.randint(0, 9999999):07d}" + rng.choice(["001", "002", "016", "031", "037", "048", "063", "072", "100", "121"])
    return first10 + piva_check_digit(first10)


def validate_piva(p: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", p)) and piva_check_digit(p[:10]) == p[10]


# ----------------------------------------------------------------- others

def phone_it(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.6:  # mobile
        num = f"3{rng.randint(20, 99)}{rng.randint(1000000, 9999999)}"
        style = rng.choice(["{a} {b} {c}", "{a}{b}{c}", "+39 {a} {b} {c}", "+39{a}{b}{c}", "{a}-{b}{c}"])
        return style.format(a=num[:3], b=num[3:6], c=num[6:])
    prefix = rng.choice(["02", "06", "011", "051", "055", "035", "045", "049", "081", "091", "070"])
    rest = f"{rng.randint(100000, 9999999)}"
    return rng.choice([f"{prefix} {rest}", f"{prefix}{rest}", f"+39 {prefix} {rest}", f"{prefix}/{rest}"])


def plate_it(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPRSTVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(0, 999):03d}{rng.choice(letters)}{rng.choice(letters)}"


def carta_identita(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"C{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(0, 99999):05d}{rng.choice(letters)}{rng.choice(letters)}"


def passaporto(rng: random.Random) -> str:
    return f"Y{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(0, 9999999):07d}"


def email_for(first: str, last: str, rng: random.Random) -> str:
    f = re.sub(r"[^a-z]", "", first.lower().replace("à", "a").replace("è", "e").replace("é", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u"))
    l = re.sub(r"[^a-z]", "", last.lower().replace("à", "a").replace("è", "e").replace("é", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u"))
    local = rng.choice([f"{f}.{l}", f"{f[0]}.{l}", f"{l}.{f}", f"{f}{l}{rng.randint(1, 99)}", f"{f}_{l}"])
    return f"{local}@{rng.choice(['gmail.com', 'libero.it', 'outlook.it', 'yahoo.it', 'pec.it', 'tim.it', 'virgilio.it', 'icloud.com'])}"


def format_date_it(d: date, rng: random.Random | None = None) -> str:
    months = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
              "settembre", "ottobre", "novembre", "dicembre"]
    styles = [f"{d.day:02d}/{d.month:02d}/{d.year}", f"{d.day}/{d.month}/{d.year}", f"{d.day} {months[d.month - 1]} {d.year}",
              f"{d.day:02d}.{d.month:02d}.{d.year}", f"{d.day:02d}-{d.month:02d}-{d.year}"]
    if rng is None:
        return styles[0]
    return rng.choice(styles)
