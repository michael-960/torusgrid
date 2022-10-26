raise ImportError('the history module is currently not maintained')

from michael960lib.common import IllegalActionError, ModifyingReadOnlyObjectError


class EvolverHistory:
    """
    A history object
    """
    def __init__(self):
        self.evolver_states = []
        self.state_functions = []
        self.committed = False 
        self.final_field_state = None

    def append_state_function(self, evolver_state: dict, sf: StateFunction):
        if self.committed:
            raise ModifyingReadOnlyObjectError(
            f'history object is already committed and hence not editable', self)

        self.state_functions.append(sf)

    def commit(self, label:str, field: ComplexField2D):
        if self.committed:
            raise IllegalActionError(f'history object (label=\'{self.label}\') is already committed')

        self.final_field_state = field.export_state()
        self.label = label
        self.committed = True 

    def is_committed(self):
        return self.committed

    def get_state_functions(self) -> List[StateFunction]:
        return self.state_functions

    def get_final_field_state(self):
        if not self.committed:
            raise IllegalActionError('cannot get final state from uncommitted evolver history')
        return self.final_field_state

    def get_label(self):
        return self.label

    def export(self) -> dict:
        if self.committed:
            state = dict()
            state['label'] = self.label
            state['state_functions'] = [sf.export() for sf in self.state_functions]
            return state
        else:
            raise IllegalActionError(
            'history object (label=\'{self.label}\') has not been committed and is therefore not ready to be exported')

