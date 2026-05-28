import math
import gzip
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

_fscores = None

def readFragmentScores(name='fpscores'):
    global _fscores
    if _fscores is not None:
        return
    data = pickle.load(gzip.open('%s.pkl.gz' % name, 'rb'))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    """
    Calculate synthetic accessibility score similar to Ertl's method.
    Returns a value between 1 (difficult) and 10 (easy).
    If fragment score file is not available, falls back to a heuristic.
    """
    global _fscores
    try:
        if _fscores is None:
            # try to load fpscores.pkl.gz in current directory
            try:
                readFragmentScores('fpscores')
            except Exception:
                # no fragment scores available, fallback to heuristic
                return _fallback_sa_score(m)

        # fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += _fscores.get(sfp, -4) * v
        score1 /= nf if nf else 1.0

        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        ri = m.GetRingInfo()
        nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = math.pow(nAtoms, 1.005) - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # transform to 1..10
        min_val = -4.0
        max_val = 2.5
        sascore = 11. - (sascore - min_val + 1) / (max_val - min_val) * 9.
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        if sascore < 1.:
            sascore = 1.0

        return float(sascore)
    except Exception:
        return _fallback_sa_score(m)

def _fallback_sa_score(m):
    """Simple heuristic fallback SA: heavier/complex molecules get lower scores."""
    try:
        nAtoms = m.GetNumAtoms()
        ringCount = Chem.rdMolDescriptors.CalcNumRings(m)
        chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        score = 8.0
        if nAtoms > 50:
            score -= (nAtoms - 50) * 0.05
        if ringCount > 4:
            score -= (ringCount - 4) * 0.5
        score -= chiral * 0.2
        return max(1.0, min(10.0, score))
    except:
        return 3.0








