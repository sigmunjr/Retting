import fnmatch
import glob
import zipfile
import os, shutil
import nose
from nose_htmloutput import HtmlOutput
import subprocess

ADDITIONAL_ENVIRONMENT_VARIABLES = '/home/sigmund/anaconda2/bin'


def copytree(src, dst, symlinks=False, ignore=None):
  for root, dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    dst_path = os.path.join(dst, rel)
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
    for f in files:
      filepath = os.path.join(dst_path, f)
      if not os.path.exists(filepath):
        shutil.copyfile(os.path.join(root, f), filepath)


def get_last_deliveries(deliveries_root):
  last_deliveries = []
  for root, dirs, files in os.walk(deliveries_root, topdown=True):
    deliveries = fnmatch.filter(dirs, 'delivery*')
    if len(deliveries)>0:
      last_delivery = sorted(deliveries)[-1:]
      dirs[:] = last_delivery
      last_deliveries += [os.path.join(root, last_delivery[0])]
  return last_deliveries

def convert_notebooks_to_html(deliveries):
  for root in deliveries:
    exist = True
    notebooks = glob.glob('*.ipynb')
    for n in notebooks:
      exist = exist and os.path.exists(n.replace('.ipynb', '.html'))
    if exist: continue

    old_cwd = os.getcwd()
    os.chdir(root)
    print subprocess.call(['jupyter nbconvert --to html *.ipynb'], shell=True)
    print 'DIR', os.getcwd()
    os.chdir(old_cwd)


def run_nose_tests(deliveries):
  for root in deliveries:
    old_cwd = os.getcwd()
    os.chdir(os.path.join(root))
    print subprocess.call(['nosetests', '--with-html'])
    print 'DIR', os.getcwd()
    os.chdir(old_cwd)


def open_browser(deliveries, python_file_depth=4, text_editor='atom'):
  import re
  import webbrowser, itertools
  student_id_pattern = re.compile('.*\\(groupid\\=(\\d+)\\).*')
  for root in deliveries:
    student_id = student_id_pattern.match(root).group(1)
    webbrowser.open_new('https://devilry.ifi.uio.no/devilry_examiner/singlegroupoverview/'+student_id)
    html_files = glob.glob(root+'/*.html')
    for h in html_files:
      webbrowser.open_new_tab(h)
    python_files = list(itertools.chain.from_iterable([glob.glob(root + '/*'*depth + '/*.py') for depth in range(python_file_depth)]))
    subprocess.call([text_editor] + python_files)
    print root
    raw_input('Press any key to load next student')

def main():
  os.environ['PATH'] += os.pathsep + ADDITIONAL_ENVIRONMENT_VARIABLES
  deliveries = get_last_deliveries(deliveries_root='/home/sigmund/Utvikling/Retting/')
  deliveries_extract = extract(deliveries)
  add_dataset(deliveries_extract, dataset_path='/home/sigmund/Utvikling/Phd/INF5860_Oblig1_solutions/code/datasets/')
  add_tests(deliveries_extract, tests_path='/tests/')

  convert_notebooks_to_html(deliveries_extract)
  run_nose_tests(deliveries_extract)
  open_browser(deliveries_extract)
  print 'DONE'


def add_tests(deliveries, tests_path):
  tests = glob.glob(tests_path+'test_*.py')
  for root in deliveries:
    for t in tests:
      shutil.copy(t, root)

def add_dataset(deliveries, dataset_path, dataset_target='/code/datasets/'):
  for root in deliveries:
    copytree(dataset_path, root + dataset_target)


def extract(delivieries, extract_name='oblig1'):
  extraction_paths = []
  for root in delivieries:
    extract_path = os.path.join(root, extract_name)
    extraction_paths.append(extract_path)
    if os.path.exists(extract_path): continue
    for f in os.listdir(root):
      filename = root + '/' + f
      if zipfile.is_zipfile(root + '/' + f):
        zf = zipfile.ZipFile(root + '/' + f, 'r')
        zf.extractall(extract_path)
  return extraction_paths


if __name__ == '__main__':
    main()