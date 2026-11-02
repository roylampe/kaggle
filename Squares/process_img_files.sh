for n in {1..324}; do
    python3 image_to_csv.py images/n-$n.svg --csv image_submission.csv --n $n --append
done