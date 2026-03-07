from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def fetch_all_entries_once(date):

    url = "https://www.marugameboat.jp/s_pdf/date.htm"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        tables = driver.find_elements(By.TAG_NAME, "table")

        all_entries = []
        race_no = 0

        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            temp_entries = []

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")

                if len(cols) >= 6:

                    lane = cols[0].text.strip()
                    if not lane.isdigit():
                        continue

                    raw_info = cols[1].text.strip()

                    # 登録番号（4桁）
                    racer_no = raw_info[:4]

                    # 登録番号以降
                    raw_name_part = raw_info[4:].strip()

                    # 通常パターン（苗字　名前　支部）
                    if "　" in raw_name_part:
                        parts = raw_name_part.split("　")

                        if len(parts) >= 3:
                            surname = parts[0]
                            firstname = parts[1]
                            branch = parts[2]
                        else:
                            surname = parts[0]
                            firstname = parts[1]
                            branch = ""

                        full_name = surname + firstname

                    else:
                        # スペース無し例外（例：長野瀬商工）
                        full_name = raw_name_part[:-2]
                        branch = raw_name_part[-2:]

                    name_branch = f"{full_name} {branch}"

                    grade = cols[2].text.strip()
                    fl = cols[3].text.strip()

                    # 🔥 修正済み（ズレ解消）
                    win_rate = cols[4].text.strip()
                    quinella_rate = cols[5].text.strip()

                    temp_entries.append({
                        "lane": int(lane),
                        "racer_no": racer_no,
                        "name_branch": name_branch,
                        "grade": grade,
                        "fl": fl,
                        "win_rate": win_rate,
                        "quinella_rate": quinella_rate
                    })

            # 6人揃ったテーブルだけ採用
            if len(temp_entries) == 6:
                race_no += 1

                for entry in temp_entries:
                    entry["race_no"] = race_no
                    all_entries.append(entry)

        return all_entries

    finally:
        driver.quit()
