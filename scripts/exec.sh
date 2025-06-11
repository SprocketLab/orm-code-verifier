#! /bin/bash
set -e


# FILES="${1}/gsm8k/*"
# for file in $FILES; do  

#   echo $file 
#   if [ -f "${file}" ]; then   # Check whether file exists.
#     python scripts/exec_shard.py --num_workers=16 --output_dir="${1}_exec" gsm8k "${file}"
    
#   fi
# done


FILES="${1}/code_contests/*"
for file in $FILES; do  

  echo $file 
  if [ -f "${file}" ]; then   # Check whether file exists.
    python scripts/exec_shard.py --num_workers=16 --output_dir="${1}_exec" code_contests "${file}" --no_generated
    
  fi
done

