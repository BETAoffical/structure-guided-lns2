import unittest
from scripts import register_sa_crossfit_report_fix as fix


class ReportFixTests(unittest.TestCase):
    def test_scope_excludes_execution_and_training(self):
        before=fix.OLD_VERIFY+"\nassert "+fix.OLD_PAIR+"\nWORKERS=20\n"
        after=before.replace(fix.OLD_VERIFY,fix.NEW_VERIFY).replace(fix.OLD_PAIR,fix.NEW_PAIR)
        fix.prove_scope(before,after)
        with self.assertRaises(ValueError):fix.prove_scope(before,after.replace("WORKERS=20","WORKERS=4"))

    def test_official_has_no_actor_keys_but_all_scientific_fields_remain(self):
        event=dict(action={"mode":"official"},delta={"paths":[[0,1]]},decision=1,uniform=.1,metrics={"runtime":1})
        new=dict(event,metrics={"runtime":2})
        self.assertTrue(eval(fix.NEW_PAIR,{},dict(event=event,new=new)))
        for key,value in (("delta",{"paths":[[0,2]]}),("uniform",.2),("decision",2),("action",{"mode":"explicit"})):
            self.assertFalse(eval(fix.NEW_PAIR,{},dict(event=event,new=dict(new,**{key:value}))))

    def test_behavior_change_requires_identical_pre_action_conditions(self):
        a=dict(before="hash",action={"agents":[1]},pool=[1,2],features=[0,1],selection_draw=.5,
               decision=0,selected_id="a",probabilities={"a":.55,"b":.45},delta={"paths":[[1]]})
        b=dict(a,action={"agents":[2]},selected_id="b",probabilities={"a":.45,"b":.55})
        same,first=fix.first_difference([a],[b])
        self.assertEqual(same,0)
        self.assertAlmostEqual(first["total_variation"],.1)
        self.assertEqual(fix.first_difference([a],[a]),(1,None))
        with self.assertRaises(ValueError):fix.first_difference([a],[dict(b,before="changed")])
        with self.assertRaises(ValueError):fix.first_difference([a],[dict(a,delta={})])


if __name__=="__main__":unittest.main()
