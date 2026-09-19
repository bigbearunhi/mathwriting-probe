"""Tolerance reaches multiprocessing conversion and changes compression, not labels."""
import io,json,sqlite3,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
import numpy as np
from ctc_data import curve_features

class ToleranceTests(unittest.TestCase):
    def test_cli_compression_and_metadata(self):
        u=np.linspace(0,1,81)
        points=np.column_stack([u,.25*np.sin(5*np.pi*u),u])
        ink=('<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="normalizedLabel">x</annotation><trace>'+','.join(' '.join(map(str,p)) for p in points)+'</trace></ink>').encode()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);archive=base/'input.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                entry=tarfile.TarInfo('train/example.inkml');entry.size=len(ink);tar.addfile(entry,io.BytesIO(ink))
            counts=[]
            for tol in (.001,.05):
                out=base/str(tol)
                r=subprocess.run([sys.executable,'ctc_data.py','--archive',str(archive),'--out',str(out),'--workers','2','--tolerance',str(tol)],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                report=json.loads((out/'preparation.json').read_text());self.assertEqual(report['tolerance'],tol)
                with sqlite3.connect(out/'features.sqlite') as db:
                    label,n,blob=db.execute('SELECT label,length,features FROM samples').fetchone()
                self.assertEqual(label,'x');x=np.frombuffer(blob,dtype='<f4').reshape(n,10)
                self.assertTrue(np.isfinite(x).all());np.testing.assert_allclose(x[:,:2].sum(0),[2,0],atol=1e-5);counts.append(n)
            self.assertGreater(counts[0],counts[1])
    def test_rejects_nonpositive_or_nonfinite_tolerance(self):
        for tol in (0,-1,float('nan'),float('inf')):
            with self.subTest(tolerance=tol),self.assertRaises(ValueError):
                curve_features([[[0,0,0],[1,1,1]]],tolerance=tol)

if __name__=='__main__':unittest.main()
