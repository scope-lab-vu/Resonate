from pathlib import Path

for line in Path('tmp.txt').read_text().strip().split():
    line = Path(line).stem.split('_')[:2]
    num = line[1]
    line = '_'.join(line) + '.xml'

    command = 'source scripts/run_evaluation.sh /home/bradyzhou/code/leaderboard/data/routes/%s %s.txt' % (line, num)

    print(command)
