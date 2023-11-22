import os

# Cesta, ktorú chcete skontrolovať
path_to_check = 'Dataset/paragraphs/valid/0/00401.jpg_p1.png'

# Skontrolujte, či cesta existuje
if os.path.exists(path_to_check):
    print("Cesta existuje.")
else:
    print("Cesta neexistuje. Skontrolujte, či ste v správnom pracovnom priečinku a či cesta k súboru je správna.")

# Skontrolujte, či je cesta súborom
if os.path.isfile(path_to_check):
    print("Je to súbor.")
else:
    print("Daná cesta nie je súbor. Uistite sa, že ste urobili správne odvolanie na súbor.")
