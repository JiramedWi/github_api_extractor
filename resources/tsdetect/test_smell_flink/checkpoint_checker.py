# pandas as pd
import pandas as pd

# read json file to pandas
file_path = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/checkpoint_flink.json"

df = pd.read_json(file_path)
df.tail(10)
print(df)

