#!/bin/bash

hive_sha="../../hive_use_for_run_java.pkl"
tsdetect="../TestSmellDetector.jar"
hive_repo="/Users/Jumma/github_repo/hive"
directory_path="/Users/Jumma/github_repo/hive"

#TODO Write bash to find @Test in the file
is_test_file() {
  local file="$1"
  if grep -q "@Test" "$file"; then
    return 0
  else
    return 1
  fi
}

collect_test_files() {
    local root_dir="$1"
    local test_files=()

    while IFS= read -r -d '' file; do
        if is_test_file "$file"; then
            test_files+=("$file")
        fi
    done < <(find "$root_dir" -type f -name "*.java" -print0)
    printf '%s\n' "${test_files[@]}"
}

write_file_to_use_in_Jar() {
    local project="$1"
    local testcase_files=($(collect_test_files "$directory_path"))
    for file in "${testcase_files[@]}"; do
        printf "%s,%s\n" "$project" "$file"
    done
}

auto_checkout() {
    sha_opened=($(python3 -c "import pandas as pd; from pathlib import Path; df = pd.read_pickle(Path('$hive_sha')); print(' '.join(df['opened']))"))
    sha_closed=($(python3 -c "import pandas as pd; from pathlib import Path; df = pd.read_pickle(Path('$hive_sha')); print(' '.join(df['closed']))"))

    for ((count=0; count < ${#sha_opened[@]}; count++)); do
        url=$(python3 -c "import pandas as pd; from pathlib import Path; df = pd.read_pickle(Path('$hive_sha')); print(df['url'][0])")

        echo "$url"
        echo "${sha_opened[count]}"
        git -C "$hive_repo" checkout "${sha_opened[count]}"
        git -C "$hive_repo" fetch
        sleep 5
        echo 'checked opened'
        write_file_to_use_in_Jar 'hive' > "../opened_hive_file_${count}_${sha_opened[count]}.csv"
        sleep 3
        java -jar "$tsdetect" "../opened_hive_file_${count}_${sha_opened[count]}.csv"
        sleep 2
        echo 'Done open'

        echo "${sha_closed[count]}"
        git -C "$hive_repo" checkout "${sha_closed[count]}"
        git -C "$hive_repo" fetch
        sleep 5
        echo 'checked closed'
        write_file_to_use_in_Jar 'hive' > "../closed_hive_file_${count}_${sha_closed[count]}.csv"
        sleep 3
        java -jar "$tsdetect" "../closed_hive_file_${count}_${sha_closed[count]}.csv"
        sleep 2
        echo 'Done closed'
    done

    echo 'Done process'
}

auto_checkout