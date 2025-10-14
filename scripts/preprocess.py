import argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DESCRIPTION")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p
def preprocess():
    train=pd.read_csv('../data/titanic/train.csv')
    test=pd.read_csv('../data/titanic/test.csv')
    train.drop(columns=['Cabin'],inplace=True)
    train.isnull().sum()
    train['Embarked'].fillna('S',inplace=True)
    test['Fare'].fillna(test['Fare'].mean(), inplace=True)    
    test.drop(columns=['Cabin'],inplace=True)


    pass
def main():
    args = build_parser().parse_args()
    # TODO: implement step
    # read args.input, write args.output

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Read input
    with args.input.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(processed_text)
if __name__ == "__main__":
    main()