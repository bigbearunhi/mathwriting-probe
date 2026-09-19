import unittest
from inspect_error import select_records, token_diff
class InspectorTests(unittest.TestCase):
 def test_select_id_and_confusion(self):
  rows=[dict(id='a',truth='x+1',prediction='X+1'),dict(id='b',truth='y',prediction='')]
  self.assertEqual(select_records(rows,sample_id='b')[0]['id'],'b')
  self.assertEqual(select_records(rows,reference='x',prediction='X')[0]['id'],'a')
 def test_diff_and_html_escaping(self):
  s=token_diff('<x','<X')
  self.assertIn('&lt;',s);self.assertIn('<del>',s);self.assertIn('<ins>',s)
 def test_correct_sample_retained(self):
  rows=[dict(id='ok',truth='x',prediction='x')]
  self.assertEqual(select_records(rows)[0]['id'],'ok')
 def test_missing_id(self):
  self.assertEqual(select_records([],sample_id='missing'),[])
if __name__=='__main__':unittest.main()
