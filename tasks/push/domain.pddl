; Domain file: tabletop-push

(define (domain tabletop-push)
  (:requirements :strips :typing)
  (:types
    block
    table
  )
  (:predicates
    (block   ?b)           ; ?b is a block
    (table   ?t)           ; ?t is a table
    (on-table ?b ?t)       ; block ?b is on table ?t
    (red    ?b)
    (blue   ?b)
    (green  ?b)
  )

  ;; PUSH: pushes a block directly onto a table
  (:action push
    :parameters (?b - block ?t - table)
    :precondition (and (block ?b) (table ?t))
    :effect (on-table ?b ?t)
  )
)
