from bs4 import BeautifulSoup

def parse_entry(html):

    soup = BeautifulSoup(html, "html.parser")

    rows = soup.select("table tr")

    entry = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        lane = cols[0].get_text(strip=True)
        if not lane.isdigit():
            continue

        entry.append({
            "lane": lane,
            "racer": cols[1].get_text(strip=True)
        })

    return entry
