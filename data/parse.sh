python build_relation_database.py --dir_path stac
cp stac/relation_database.json molweni/relation_database.json

python parse.py --dir_path stac --num_contexts 20 $1
python parse.py --dir_path molweni --num_contexts 14 $1

python parse_test.py --dir_path stac --num_contexts 37 $1 --mode dev $1
python parse_test.py --dir_path stac --num_contexts 37 $1
python parse_test.py --dir_path molweni --num_contexts 14 --mode dev $1
python parse_test.py --dir_path molweni --num_contexts 14 $1
