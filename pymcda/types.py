"""
.. module:: types
.. moduleauthor:: Olivier Sobrie <olivier@sobrie.be>
"""

from __future__ import division, print_function
import random
import sys
from itertools import product
from functools import cmp_to_key
from xml.etree import ElementTree
from copy import deepcopy
from collections import OrderedDict

type2tag = {
    int: 'integer',
    float: 'real',
    str: 'label',
}

unmarshallers = {
    'integer': lambda x: int(x.text),
    'real': lambda x: float(x.text),
    'label': lambda x: str(x.text),
}

def marshal(value):
    tag = type2tag.get(type(value))
    e = ElementTree.Element(tag)
    e.text = str(value)
    return e

def unmarshal(xml):
    m = unmarshallers.get(xml.tag)
    return m(xml)

def find_xmcda_tag(xmcda, tag, id = None):
    if xmcda.tag == tag and (id is None or xmcda.get('id') == id):
        return xmcda

    if id is None:
        search_str = ".//%s" % tag
    else:
        search_str = ".//%s[@id='%s']" % (tag, id)

    xmcda = xmcda.find(search_str)
    if xmcda is None:
        raise TypeError("%s::invalid tag" % tag)

    return xmcda

class McdaDict(object):
    """This class allows to declare an MCDA dictionnary element.
    It contains usefull methods to manipulate MCDA data.
    """

    def __init__(self, l = list(), id = None):
        """Create a new MCDA dictionnary instance

        Kwargs:
           l (list): A list containing a set of MCDA objects
           id (str): The identifier of the dictionnary
        """
        self.id = id
        self._d = OrderedDict()
        for i in l:
            self._d[i.id] = i

    def __contains__(self, key):
        """Check if a MCDA object is present in the dictionnary. If the
        object is contained in the dictionnary, True is returned,
        otherwise False is returned."""

        return self._d.__contains__(key)

    def __eq__(self, mcda_dict):
        """This method allows to compare the current MCDA dictionnary to
        another MCDA dictionnary. If the two MCDA dictionnaries are
        equal, then True is returned, otherwise False is returned."""

        return self._d.__eq__(dict(mcda_dict._d))

    def __hash__(self):
        return hash(frozenset(self._d.iteritems()))

    def __iter__(self):
        """Return an iterator object for the MCDA dictionnary."""

        try:
            return self._d.itervalues()
        except:
            return iter(self._d.values())

    def __getitem__(self, key):
        """Lookup for an MCDA object in the dictionnary on basis of its
        identifier. If there is no object with idenfier 'id' in the
        MCDA dictionnary, then KeyError exception is returned."""

        return self._d.__getitem__(key)

    def __len__(self):
        """Return the number of elements contained in the MCDA
        dictionnary."""

        return self._d.__len__()

    def append(self, mcda_object):
        """Add an MCDA object in the MCDA dictionnary"""

        self._d[mcda_object.id] = mcda_object

    def copy(self):
        """Perform a full copy of the MCDA dictionnary"""

        return deepcopy(self)

    def has_key(self, key):
        """Check if MCDA object id is in the dictionnary"""

        return self._d.has_key(key)

    def items(self):
        """Return a copy of the MCDA dictionary's list of (id, mcda_object)
        pairs"""

        return self._d.items()

    def iterkeys(self):
        """Return an iterator over the MCDA dictionary's object IDs"""

        return self._d.iterkeys()

    def itervalues(self):
        """Return an iterator over the MCDA dictionary's object IDs"""

        return iter(self._d.values())

    def keys(self):
        """Return the list of MCDA object IDs contained in the MCDA
        dictionnary"""

        return self._d.keys()

    def remove(self, key):
        """This method allows to remove an element from the dictionnary"""

        del self._d[key]

    def update(self, mcda_dict):
        """Add the object of a second MCDA dictionnary into the current
        dictionnary"""

        self._d.update(mcda_dict)

    def values(self):
        """Return the list of MCDA objects contained in the MCDA
        dictionnary"""

        return self._d.values()

    def to_list(self):
        """Return a list of MCDA objects contained in the MCDA dictionnary
        ordered by MCDA object ID"""

        l = self._d.values()
        l.sort(key = lambda x: x.id)
        return l

    def get_subset(self, ids):
        """Return a subset of the current MCDA dictionnary containing the
        MCDA object IDs"""

        return type(self)([self._d[id] for id in ids])

    def split(self, n, proportions = None, randomize = True):
        """Split the MCDA dictionnary into two or several parts

        Kargs:
           n (integer): Number of parts in which the MCDA dictionnary
                        should be split

        Kwargs:
           propostions (list): A list containing the proportions of
                               subset
           randomize (bool): Wheter or not the split should be randomize
        """

        if proportions is None:
            proportions = [1 / n] * n
        elif len(proportions) == n:
            t = sum(proportions)
            proportions = [proportion / t for proportion in proportions]
        else:
            raise ValueError('%s::split invalid proportions')

        keys, nkeys = list(self._d.keys()), len(self._d.keys())
        j, subsets = 0, []

        if randomize is True:
            random.shuffle(keys)

        for proportion in proportions:
            j2 = int(j + proportion * nkeys)
            subset = type(self)(self._d[i] for i in keys[j:j2])
            j = j2
            subsets.append(subset)

        subset = type(self)(self._d[i] for i in keys[j2:nkeys])
        subsets[-1].update(subset)

        return tuple(subsets)


class McdaObject(object):

    def __eq__(self, other):
        """Return True if the two MCDA objects are equal, otherwise
        return False"""

        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.id)

    def copy(self):
        """Return a copy of the current MCDA object"""

        return deepcopy(self)

class CriteriaSets(object):

    def __init__(self, cs = None, id = None):
        self.id = id
        self.cs = cs if cs is not None else set()

    def __iter__(self):
        """Return an iterator object for the MCDA dictionnary."""

        return self.cs.__iter__()

    def __len__(self):
        return len(self.cs)

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "CriteriaSets(%s)" % ', '.join(map(str, self.cs))

    def add(self, cs):
        return self.cs.add(cs)

    def copy(self):
        """Perform a full copy of the Criteria set"""

        return deepcopy(self)

    def remove(self, x):
        return self.cs.remove(x)

    def discard(self, x):
        return self.cs.discard(x)

    def to_xmcda(self, id = None):
        """Convert the MCDA dictionnary into XMCDA output"""

        root = ElementTree.Element('CriteriaSets')

        if id is not None:
            root.set('id', id)
        elif self.id is not None:
            root.set('id', self.id)

        for cs in self:
            xmcda = cs.to_xmcda()
            root.append(xmcda)

        return root

    def from_xmcda(self, xmcda, id = None):
        """Read the MCDA dictionnary from XMCDA input"""

        xmcda = find_xmcda_tag(xmcda, 'criteriaSets', id)

        self.id = xmcda.get('id')

        tag_list = xmcda.getiterator('criteriaSet')
        for tag in tag_list:
            cs = CriteriaSet().from_xmcda(tag)
            self.add(cs)

        return self

class CriteriaSet(object):

    def __init__(self, criteria = None):
        self.criteria = set(criteria) if criteria is not None else set()

    def __repr__(self):
        return "CriteriaSet(%s)" % ', '.join(map(str, self.criteria))

    def __eq__(self, other):
        return set(self.criteria) == set(other.criteria)

    def __hash__(self):
        return hash(frozenset(self.criteria))

    def __iter__(self):
        return self.criteria.__iter__()

    def __len__(self):
        return len(self.criteria)

    def __str__(self):
        return '[%s]' % ', '.join(map(str, self.criteria))

    def add(self, x):
        return self.criteria.add(x)

    def copy(self):
        """Return a copy of the current MCDA object"""

        return deepcopy(self)

    def remove(self, x):
        return self.criteria.remove(x)

    def discard(self, x):
        return self.criteria.discard(x)

    def issubset(self, criteria):
        if isinstance(criteria, CriteriaSet) is False \
           and isinstance(criteria, set) is False \
           and isinstance(criteria, frozenset) is False:
            return criteria in self.criteria

        return self.criteria.issubset(criteria)

    def issuperset(self, criteria):
        if isinstance(criteria, CriteriaSet) is False \
           and isinstance(criteria, set) is False \
           and isinstance(criteria, frozenset) is False:
            return criteria in self.criteria

        return self.criteria.issuperset(criteria)

    def to_xmcda(self):
        """Convert the CriteriaSet into XMCDA"""

        root = ElementTree.Element('criteriaSet')
        for c in self.criteria:
            el = ElementTree.SubElement(root, 'element')
            cid = ElementTree.SubElement(el, 'criterionID')
            cid.text = str(c)

        return root

    def from_xmcda(self, xmcda):
        """Read the MCDA object from XMCDA input"""
        xmcda = find_xmcda_tag(xmcda, 'criteriaSet')

        tag_list = xmcda.getiterator('criterionID')
        for tag in tag_list:
            self.criteria.add(tag.text)

        return self

class Criteria(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "Criteria(%s)" % self.values()

    def get_active(self):
        return Criteria([c for c in self if c.disabled is not True])

    def to_xmcda(self, id = None):
        """Convert the MCDA dictionnary into XMCDA output"""

        root = ElementTree.Element('criteria')

        if id is not None:
            root.set('id', id)
        elif self.id is not None:
            root.set('id', self.id)

        for crit in self:
            crit_xmcda = crit.to_xmcda()
            root.append(crit_xmcda)

        return root



class Criterion(McdaObject):

    MINIMIZE = -1
    MAXIMIZE = 1
    SINGLEPEAKED = 2
    SINGLEVALLEYED = -2

    def __init__(self, id=None, name=None, disabled=False,
                 direction=MAXIMIZE, weight=None, thresholds=None, dupl_id=None):
        """Create a new Criterion instance

        Kwargs:
           id (str): Identifier of the criterion
           name (str): A friendly name for the criterion
           disabled (bool): Whether or not this criterion is disabled
           direction (integer): Equal to -1 if criterion is to minimize,
                                1 if the criterion is to maximize
           weight (float): Deprecated
           thresholds (list): List of threshold associated to the criterion
        """

        self.id = id
        self.dupl_id = dupl_id
        self.name = name
        self.disabled = disabled
        self.direction = direction
        self.weight = weight
        self.thresholds = thresholds

    def __repr__(self):
        """Manner to represent the MCDA object"""
        if self.direction == 1:
            direction = "+"  
        elif self.direction == -1:
            direction = "-"
        elif self.direction == 2:
            direction = "+-"
        elif self.direction == -2:
            direction = "-+"
        else:
            direction = "unknowned"
            
        return "%s (%s)" % (self.id, direction)


class CriteriaValues(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "CriteriaValues(%s)" % self.values()

    def __str__(self):
        l = 0
        for cv in self:
            l = max(len("%s" % cv.id), l)

        string = ""
        for cv in self:
            string += "%*s: %s\n" % (l, cv.id, cv.value)

        return string[:-1]

    def min(self):
        return min([cv.value for cv in self])

    def max(self):
        return max([cv.value for cv in self])

    def sum(self):
        return sum([cv.value for cv in self])

    def normalize(self, vmin = None, vmax = None):
        vmin = self.min() if vmin is None else vmin
        vmax = self.max() if vmax is None else vmax

        for cv in self:
            cv.value = (cv.value - vmin) / (vmax - vmin)

    def normalize_sum_to_unity(self):
        """Method that allow to  normalize all the criteria values
        contained in the MCDA dictionnary"""

        total = sum([cv.value for cv in self])

        for cv in self:
            cv.value /= total


class CriterionValue(McdaObject):

    def __init__(self, id=None, value=None):
        """Create a new CriterionValue instance

        Kwargs:
           id (str): Identifier of the Criterion
           value (float): The value associated to the criterion
        """

        self.id = id
        self.value = value

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "CriterionValue(%s: %s)" % (self.id, self.value)

    def __str__(self):
        return "%s: %s" % (self.id, self.value)

    def id_issubset(self, ids):
        if isinstance(self.id, CriteriaSet) or isinstance(self.id, set) \
           or isinstance(self.id, frozenset):
            return self.id.issubset(ids)
        elif isinstance(ids, CriteriaSet) or isinstance(ids, set) \
           or isinstance(ids, frozenset):
            if isinstance(self.id, str):
                return self.id in ids
            else:
                return ids.issuperset(self.id)
        else:
            return self.id is ids


class Alternatives(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "Alternatives(%s)" % self.values()

    def to_xmcda(self, id = None):
        """Convert the MCDA dictionnary into XMCDA output"""

        root = ElementTree.Element('alternatives')

        if id is not None:
            root.set('id', id)
        elif self.id is not None:
            root.set('id', self.id)

        for action in self:
            alt = action.to_xmcda()
            root.append(alt)

        return root


class Alternative(McdaObject):

    def __init__(self, id=None, name=None, disabled=False):
        """Create a new Alternative instance

        Kwargs:
           id (str): Identifier of the alternative
           name (str): A friendly name for the alternative
           disabled (bool): Whether or not this alternative is disabled
        """

        self.id = id
        self.name = name
        self.disabled = disabled

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "Alternative(%s)" % self.id


class PerformanceTable(McdaDict):

    def __call__(self, id):
        return self[id].performances

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "PerformanceTable(%s)" % self.values()

    def __str__(self):
        l = 0 if self.id is None else len(self.id)
        col = OrderedDict()
        for ap in self:
            l = max(len(ap.id), l)
            for k, v in ap.performances.items():
                # m = max(len(k), len("%s" % v))
                m = max(len(k), len("%s" % v.__str__()))
                col[k] = m if k not in col else max(col[k], m)

        crit = list(col.keys())

        string = "%*s" % (l, self.id) if self.id is not None else " " * l

        for c in crit:
            string += " %*s" % (col[c], c)

        string += "\n"
        for ap in self:
            string += "%*s" % (l, ap.id)
            for c in crit:
                if c in ap.performances:
                    string += " %*s" % (col[c], ap.performances[c])
                else:
                    string += " " * (col[c] + 1)

            string += "\n"

        return string[:-1]

    def __mathop(self, value, op):
        out = PerformanceTable([], self.id)
        if type(value) == float or type(value) == int \
           or type(AlternativePerformances):
            for key, val in self.items():
                if op == "add":
                    ap = val + value
                elif op == "sub":
                    ap = val - value
                elif op == "mul":
                    ap = val * value
                elif op == "div":
                    ap = val / value
                out.append(ap)
        elif type(PerformanceTable):
            for key, val in self.items():
                ap = val * value[key]
                out.append(ap)
        else:
            raise TypeError("Invalid value type (%s)" % type(value))

        return out

    def __add__(self, value):
        """Add value to the performances"""

        return self.__mathop(value, "add")

    def __sub__(self, value):
        """Substract value to the performances"""

        return self.__mathop(value, "sub")

    def __mul__(self, value):
        """Multiply performances by value"""

        return self.__mathop(value, "mul")

    def __div__(self, value):
        return self.__truediv__(value)

    def __truediv__(self, value):
        """Divide performances by value"""

        return self.__mathop(value, "div")

    def get_criteria_ids(self):
        return next(self.itervalues()).performances.keys()

    def update_direction(self, c):
        """Multiply all performances by -1 if the criterion is to
        minimize"""

        for ap in self._d.values():
            ap.update_direction(c)

    def round(self, k = 3, cids = None):
        """Round all performances on criteria cids to maximum k digit

        Kwargs:
           k (int): max number of digit
           cids (list): list of criteria which should be rounded to k
                        digit
        """

        if cids is None:
            #cids = next(self._d.itervalues()).performances.keys()
            #import pdb; pdb.set_trace()
            cids = next(iter(self._d.values())).performances.keys()

        for ap in self._d.values():
            ap.round(k, cids)

    def multiply(self, value, cids = None):
        """Multiply all performance on criteria cids by value

        Kwargs:
           value (float): value by which each performance should be
                          multiplied
           cids (list): list of criteria which should be multiplied by
                        value
        """

        if cids is None:
            cids = next(self._d.itervalues()).performances.keys()

        for ap in self._d.values():
            ap.multiply(value, cids)

    def get_best(self, c):
        """Return the best possible fictive alternative performances"""

        perfs = next(self.itervalues()).performances
        wa = AlternativePerformances('best', perfs.copy())
        for ap, crit in product(self, c):
            wperf = wa.performances[crit.id] * crit.direction
            perf = ap.performances[crit.id] * crit.direction
            if wperf < perf:
                wa.performances[crit.id] = ap.performances[crit.id]
        return wa

    def get_worst(self, c):
        """Return the worst possible fictive alternative performances"""

        perfs = next(self.itervalues()).performances
        wa = AlternativePerformances('worst', perfs.copy())
        for ap, crit in product(self, c):
            # if not wa.performances:
            #     import pdb; pdb.set_trace()
            wperf = wa.performances[crit.id] * crit.direction
            perf = ap.performances[crit.id] * crit.direction
            if wperf > perf:
                wa.performances[crit.id] = ap.performances[crit.id]
        return wa

    def get_min(self):
        """Return an alternative which has the minimal performances on all
        criteria"""

        perfs = next(self.itervalues()).performances
        a = AlternativePerformances('min', perfs.copy())
        for ap, cid in product(self, perfs.keys()):
            perf = ap.performances[cid]
            if perf < a.performances[cid]:
                a.performances[cid] = perf
        return a

    def get_max(self):
        """Return an alternative which has the maximal performances on all
        criteria"""

        perfs = next(self.itervalues()).performances
        a = AlternativePerformances('max', perfs.copy())
        for ap, cid in product(self, perfs.keys()):
            perf = ap.performances[cid]
            if perf > a.performances[cid]:
                a.performances[cid] = perf
        return a

    def get_mean(self):
        """Return an alternative which has the mean performances on all
        criteria"""

        cids = next(self.itervalues()).performances.keys()
        a = AlternativePerformances('mean', {cid: 0 for cid in cids})
        for ap, cid in product(self, cids):
            perf = ap.performances[cid]
            a.performances[cid] += perf

        for cid in cids:
            a.performances[cid] /= len(self)

        return a

    def get_range(self):
        """Return the range of the evaluations on each criterion"""

        ap_min = self.get_min().performances
        ap_max = self.get_max().performances

        cids = next(self.itervalues()).performances.keys()
        a = AlternativePerformances('range')
        for cid in cids:
            a.performances[cid] = ap_max[cid] - ap_min[cid]

        return a

    def get_unique_values(self):
        """Return the unique values on each criterion"""

        cids = next(self.itervalues()).performances.keys()
        a = {cid: set() for cid in cids}
        for ap, cid in product(self, cids):
            perf = ap.performances[cid]
            a[cid].add(perf)

        for key, val in a.items():
            a[key] = sorted(list(val))

        return a


class AlternativePerformances(McdaObject):

    def __init__(self, id=None, performances=None, alternative_id=None):
        """Create a new AlternativePerformances instance

        Kwargs:
           id (str): Identifier of the alternative
           performances (dict): Alternatives' performances on the
                                different criteria
        """

        self.id = id
        self.altid = alternative_id
        if self.altid is None:
            self.altid = id

        self.performances = OrderedDict({}) if performances is None else performances

    def __call__(self, criterion_id):
        """Return the performance of the alternative on criterion_id"""
        return self.performances[criterion_id]

    def __repr__(self):
        """Manner to represent the MCDA object"""
        return "AlternativePerformances(%s: %s)" % (self.id, self.performances)

    def __str__(self):
        l = 0
        for k in self.performances.keys():
            l = max(len(k), l)

        l += len(self.id)

        string = "%s:" % self.id
        for k, v in self.performances.items():
            string += " %s: %s\n" % (k, v)
            string += " " * (len(self.id) + 1)

        return string[:string.rfind('\n')]

    def __mathop(self, value, op):
        out = AlternativePerformances(self.id)
        if type(value) == float or type(value) == int:
            for key in self.performances.keys():
                if op == 'add':
                    out.performances[key] = self.performances[key] + value
                elif op == 'sub':
                    out.performances[key] = self.performances[key] - value
                elif op == 'mul':
                    out.performances[key] = self.performances[key] * value
                elif op == 'div':
                    out.performances[key] = self.performances[key] / value
        elif type(value) == AlternativePerformances:
            for key in self.performances.keys():
                if op == 'add':
                    out.performances[key] = self.performances[key] + \
                                            value.performances[key]
                elif op == 'sub':
                    out.performances[key] = self.performances[key] - \
                                            value.performances[key]
                elif op == 'mul':
                    out.performances[key] = self.performances[key] * \
                                            value.performances[key]
                elif op == 'div':
                    out.performances[key] = self.performances[key] / \
                                            value.performances[key]
        else:
            raise TypeError("Invalid value type (%s)" % type(value))

        return out

    def __add__(self, value):
        """Add value to the performances"""

        return self.__mathop(value, "add")

    def __sub__(self, value):
        """Substract value to the performances"""

        return self.__mathop(value, "sub")

    def __mul__(self, value):
        """Multiply performances by value"""

        return self.__mathop(value, "mul")

    def __div__(self, value):
        return self.__truediv__(value)

    def __truediv__(self, value):
        """Divide performances by value"""

        return self.__mathop(value, "div")

    def update_direction(self, c):
        """Multiply all performances by -1 if the criterion is to
        minimize if monotone criteria """

        for crit in c:
            if abs(crit.direction) == 1:
                self.performances[crit.id] *= crit.direction

                
    def round(self, k = 3, cids = None):
        """Round all performances on criteria cids to maximum k digit
        Kwargs:
           k (int): max number of digit
           cids (list): list of criteria which should be rounded to k
                        digit
        """

        if cids is None:
            cids = self.performances.keys()

        for key, value in self.performances.items():
            self.performances[key] = round(value, k)

    def multiply(self, value, cids = None):
        """Multiply all performance on criteria cids by value

        Kwargs:
           value (float): value by which each performance should be
                          multiplied
           cids (list): list of criteria which should be multiplied by
                        value
        """

        if cids is None:
            cids = self.performances.keys()

        for key in self.performances:
            self.performances[key] *= value


class CategoriesValues(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "CategoriesValues(%s)" % self.values()

    def display(self, out = sys.stdout):
        for cv in self:
            cv.display(out)

    def get_upper_limits(self):
        d = {}
        for cv in self:
            d[cv.id] = cv.value.upper
        return d

    def get_lower_limits(self):
        d = {}
        for cv in self:
            d[cv.id] = cv.value.lower
        return d

    def __cmp_categories_values(self, catva, catvb):
        if catva.value.lower > catvb.value.lower:
            return 1
        elif catva.value.lower < catvb.value.lower:
            return 0
        elif catva.value.upper > catvb.value.upper:
            return 1
        else:
            return 0

    def get_ordered_categories(self):
        """Get the list of ordered categories"""

        catvs = sorted(self, key = cmp_to_key(self.__cmp_categories_values))
        return [catv.id for catv in catvs]

    def to_categories(self):
        """Convert the content of the dictionnary into Categories()"""

        cats = Categories()
        for i, cat in enumerate(reversed(self.get_ordered_categories())):
            cat = Category(cat, rank = i + 1)
            cats.append(cat)
        return cats


class CategoryValue(McdaObject):

    def __init__(self, id = None, value = None):
        self.id = id
        self.value = value

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "CategoryValue(%s: %s)" % (self.id, self.value)

    def pprint(self):
        return "%s: %s" % (self.id, self.value.pprint())

    def display(self, out = sys.stdout):
        print(self.pprint(), file = out)


class Interval(McdaObject):

    def __init__(self, lower = float("-inf"), upper = float("inf")):
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "Interval(%s,%s)" % (self.lower, self.upper)

    def pprint(self):
        return "%s - %s" % (self.lower, self.upper)

    def display(self):
        print(self.pprint())

    def included(self, value):
        if self.lower and value < self.lower:
            return False

        if self.upper and value > self.upper:
            return False

        return True


class AlternativesValues(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "AlternativesValues(%s)" % self.values()


class AlternativeValue(McdaObject):

    def __init__(self, id = None, value = None):
        self.id = id
        self.value = value

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "AlternativeValue(%s: %s)" % (self.id, self.value)



class Categories(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "Categories(%s)" % self.values()

    def get_ids(self):
        return self.keys()

    def get_ordered_categories(self):
        d = {c.id: c.rank for c in self}
        return sorted(d, key = lambda key: d[key], reverse = True)


class Category(McdaObject):

    def __init__(self, id=None, name=None, disabled=False, rank=None):
        self.id = id
        self.name = name
        self.disabled = disabled
        self.rank = rank

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "%s: %d" % (self.id, self.rank)


class Limits(McdaObject):

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper  = upper

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "Limits(%s,%s)" % (self.lower, self.upper)


class CategoriesProfiles(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "CategoriesProfiles(%s)" % self.values()

    def get_ordered_profiles(self):
        lower_cat = { cp.value.lower: cp.id for cp in self }
        upper_cat = { cp.value.upper: cp.id for cp in self }

        lowest_highest = set(lower_cat.keys()) ^ set(upper_cat.keys())
        lowest = list(set(lower_cat.keys()) & lowest_highest)

        profiles = [ lower_cat[lowest[0]] ]
        for i in range(1, len(self)):
            ucat = self[profiles[-1]].value.upper
            profiles.insert(i, lower_cat[ucat])

        return profiles

    def get_ordered_categories(self):
        profiles = self.get_ordered_profiles()
        categories = [ self[profiles[0]].value.lower ]
        for profile in profiles:
            categories.append(self[profile].value.upper)
        return categories

    def to_categories(self):
        cats = Categories()
        for i, cat in enumerate(reversed(self.get_ordered_categories())):
            cat = Category(cat, rank = i + 1)
            cats.append(cat)
        return cats


class CategoryProfile(McdaObject):

    def __init__(self, id = None, value = None):
        self.id = id
        self.value = value

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "%s: %s" % (self.id, self.value)


class AlternativesAssignments(McdaDict):

    def __call__(self, id):
        return self._d[id].category_id

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "AlternativesAssignments(%s)" % self.values()

    def __str__(self):
        l = 0
        for aa in self:
            l = max(len(aa.id), l)

        string = ""
        for aa in self:
            string += "%*s: %s\n" % (l, aa.id, aa.category_id)

        return string[:-1]

    def get_alternatives_in_category(self, category_id):
        l = []
        for aa in self:
            if aa.is_in_category(category_id):
                l.append(aa.id)

        return l

    def get_alternatives_in_categories(self, *category_ids):
        l = []
        for aa, cat in product(self, *category_ids):
            if aa.is_in_category(cat):
                l.append(aa.id)

        return l

    def to_alternatives(self):
        alternatives = Alternatives()
        for aa in self:
            alternatives.append(Alternative(aa.id))

        return alternatives

    def display(self, alternative_ids = None, out = sys.stdout):
        if alternative_ids is None:
            #alternative_ids = self.keys()
            alternative_ids = list(self.keys())
            alternative_ids.sort()

        # Compute max column length
        cols_max = {"aids": max([len(aid) for aid in alternative_ids]),
                    "category": max([len(aa.category_id)
                                    for aa in self.values()] \
                                    + [len("category")])}

        # Print header
        line = " " * (cols_max["aids"] + 1)
        line += " " * (cols_max["category"] - len("category")) \
                + "category"
        print(line, file = out)

        # Print values
        for aid in alternative_ids:
            category_id = str(self[aid].category_id)
            line = str(aid) + " " * (cols_max["aids"] - len(str(aid)) + 1)
            line += " " * (cols_max["category"] - len(category_id)) \
                    + category_id
            print(line, file = out)


class AlternativeAssignment(McdaObject):

    def __init__(self, id=None, category_id=None):
        self.id = id
        self.category_id = category_id

    def __repr__(self):
        """Manner to represent the MCDA object"""

        return "%s: %s" % (self.id, self.category_id)

    def __str__(self):
        return "%s: %s" % (self.id, self.category_id)

    def is_in_category(self, category_id):
        if self.category_id == category_id:
            return True

        return False


class Parameters(McdaDict):

    def __repr__(self):
        """Manner to represent the MCDA dictionnary"""

        return "Parameters(%s)" % self.values()


class Parameter(McdaObject):

    def __init__(self, id = None, value = None, name = None):
        self.id = id
        self.value = value
        self.name = name

    def __repr__(self):
        """Manner to represent the MCDA object"""
        return "%s: %s" % (self.id, self.value)


