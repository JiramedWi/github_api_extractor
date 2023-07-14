#!/bin/bash

#is_test_file() {
#    local filename="$1"
#    if [[ "$filename" == *.java ]]; then
#        if [[ "$filename" == *test* || "$filename" == *testcase* ]]; then
#            echo "Yes"
#        else
#            echo "No"
#        fi
#    fi
#}


hive_sha="../resources/hive_use_for_run_java.pkl"
#tsdetect="../resources/tsdetect/TestSmellDetector.jar"
hive_repo="/Users/Jumma/github_repo/hive"
directory_path="/Users/Jumma/github_repo/hive"

is_test_directory() {
    local directory="$1"
    local directory_parts=($(echo "$directory" | tr '[:upper:]' '[:lower:]' | tr '/' ' '))
    local contains_test=0
    local contains_src=0

    for part in "${directory_parts[@]}"; do
        if [ "$part" == "test" ]; then
            contains_test=1
        elif [ "$part" == "src" ]; then
            contains_src=1
        fi
    done

    if [ "$contains_test" -eq 1 ] && [ "$contains_src" -eq 1 ]; then
        return 0
    else
        return 1
    fi
}

is_test_file() {
  local file="$1"
  if grep -q "@Test" "$file"; then
    return 0
  else
    return 1
  fi
}

#TODO Write bash to find @Test in the file
#is_test_file() {
#    local filename="$1"
#    if check_java_content "$filename"; then
#      if [[ "$filename" == *.java ]]; then
#        return 0
#      else
#          return 1
#      fi
#    fi
#}

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

#collect_test_files "$directory_path"
write_file_to_use_in_Jar "hive" > "output.csv"
#write_csv() {
#  local project="$1"
#  local name="$2"
#
#  printf '%s\n' "$project" "$name"
#}
#
#write_csv "hive" "number1" > "output.csv"