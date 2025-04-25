; This is a new custom planner for a pick and place task

(define (domain tabletop-skill)
  (:requirements :strips :typing)
  (:types block)
  (:predicates
    (block ?b)                         ; ?b is a block
    (top-clear ?b)                     ; block ?b does not have anything blocking access from above
    (on-block ?b1 ?b2)                 ; block ?b1 is on top of block ?b2
    (arm-free)                         ; the robot arm is free and not holding anything
    (holding ?b)                       ; arm is currently holding block ?b.
    (red ?b)                           
    (blue ?b)                          
    (green ?b)                         
  )

  ;; PICK: picks up a block.
  (:action pick
    :parameters (?b - block)
    :precondition (and (top-clear ?b) (arm-free))
    :effect (and (holding ?b)
                 (not (arm-free)))
  )



  ;; STACK: places a block on top of another block.
  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (holding ?b1)
                       (top-clear ?b2))
    :effect (and (on-block ?b1 ?b2)
                 (not (top-clear ?b2))
                 (arm-free)
                 (not (holding ?b1)))
  )
)