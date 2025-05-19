def collect_data(output_dir: str) -> str:
    """
    Collects real estate data from cian.ru and saves it as CSV.

    Args:
        output_dir (str): Directory to save the raw data.

    Returns:
        str: Path to the saved raw CSV file.
    """
    parser = CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(output_dir, f"raw_{t}.csv")
    data = parser.get_flats(
        deal_type="sale",
        rooms=(1, 2, 3),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 20,
            "object_type": "secondary"
        }
    )
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[+] Raw data saved to {csv_path}")
    return csv_path

collect_data("data/raw")