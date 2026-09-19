import unittest
from error_types import classify_edits, check_syntax
class ErrorTypesTests(unittest.TestCase):
 def test_digit_vs_symbol_overlap(self):
  c=classify_edits('x+1','y+l')
  self.assertTrue(c['number_error']);self.assertTrue(c['symbol_error'])
 def test_brace_is_structure_not_symbol(self):
  c=classify_edits(r'\sqrt{x}',r'\sqrt{x')
  self.assertFalse(c['symbol_error']);self.assertFalse(c['number_error']);self.assertTrue(c['structure_token_error'])
 def test_valid_wrong_content(self):
  self.assertEqual(check_syntax(r'\hat{x}')['status'],'valid')
  self.assertTrue(classify_edits(r'\hat{\eta}',r'\hat{x}')['symbol_error'])
 def test_compiler_catches_more_than_braces(self):
  self.assertEqual(check_syntax('x^{2')['status'],'invalid')
  self.assertEqual(check_syntax('x^^2')['status'],'invalid')
  self.assertEqual(check_syntax(r'\frac{1}')['status'],'invalid')
 def test_safe_and_escaped_braces(self):
  self.assertEqual(check_syntax(r'\{x\}')['status'],'valid')
  self.assertEqual(check_syntax(r'\input{/etc/passwd}')['status'],'unknown')
 def test_digit_change_valid(self):
  self.assertTrue(classify_edits('12','13')['number_error'])
  self.assertEqual(check_syntax('13')['status'],'valid')

class SymbolBreakdownTests(unittest.TestCase):
 def test_confusion_deletion_insertion_and_denominator(self):
  from error_types import symbol_breakdown
  r=symbol_breakdown([{'truth':'xx+1','prediction':'Xx+l'}, {'truth':'y','prediction':''}, {'truth':'','prediction':'z'}])
  by={x['symbol']:x for x in r['symbols']}
  self.assertEqual(by['x']['reference_count'],2)
  self.assertEqual(by['x']['substituted'],1)
  self.assertEqual(by['x']['error_rate'],.5)
  self.assertEqual(by['y']['deleted'],1)
  self.assertEqual(by['z']['inserted'],1)
  self.assertIsNone(by['z']['error_rate'])
  self.assertTrue(any(x['reference']=='1' and x['prediction']=='l' for x in r['confusions']))
  self.assertNotIn('1',by)

if __name__=='__main__':unittest.main()
