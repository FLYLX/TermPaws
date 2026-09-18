import uuid
from typing import Any, List

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from pydantic import BaseModel

from app.api.deps import CurrentUser, SessionDep
from app.models import Item, ItemHandler, ItemHandlerItem, ItemHandlerUser, User

# Request body models
class AddItemToHandlerRequest(BaseModel):
    item_handler_id: uuid.UUID
    item_id: uuid.UUID

class AddUserToHandlerRequest(BaseModel):
    item_handler_id: uuid.UUID
    user_id: uuid.UUID

router = APIRouter(prefix="/item-handler-associations", tags=["item-handler-associations"])


# ItemHandlerItem routes
@router.post("/items", response_model=dict)
def add_item_to_handler(
    *, session: SessionDep, current_user: CurrentUser, request: AddItemToHandlerRequest
) -> Any:
    """
    Add an item to an item handler.
    """
    item_handler_id = request.item_handler_id
    item_id = request.item_id
    # Check if item handler exists
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check if item exists
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check permissions: user must be owner of the item handler
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to modify this item handler")
    
    # Check if the association already exists
    existing_association = session.exec(
        select(ItemHandlerItem)
        .where(ItemHandlerItem.item_handler_id == item_handler_id)
        .where(ItemHandlerItem.item_id == item_id)
    ).first()
    
    if existing_association:
        raise HTTPException(status_code=400, detail="Item is already associated with this item handler")

    # An item belongs to exactly one handler: drop any links to other
    # handlers so handler-scoped data (memory/token) resolves unambiguously.
    other_links = session.exec(
        select(ItemHandlerItem)
        .where(ItemHandlerItem.item_id == item_id)
        .where(ItemHandlerItem.item_handler_id != item_handler_id)
    ).all()
    for link in other_links:
        session.delete(link)

    # Create the association
    association = ItemHandlerItem(item_handler_id=item_handler_id, item_id=item_id)
    session.add(association)
    session.commit()

    from app.services.agent.chat_runtime import invalidate_handler_cache

    invalidate_handler_cache(str(item_id))

    return {"message": "Item added to item handler successfully"}


@router.delete("/items/{item_handler_id}/{item_id}", response_model=dict)
def remove_item_from_handler(
    *, session: SessionDep, current_user: CurrentUser, item_handler_id: uuid.UUID, item_id: uuid.UUID
) -> Any:
    """
    Remove an item from an item handler.
    """
    # Check if the association exists
    association = session.exec(
        select(ItemHandlerItem)
        .where(ItemHandlerItem.item_handler_id == item_handler_id)
        .where(ItemHandlerItem.item_id == item_id)
    ).first()
    
    if not association:
        raise HTTPException(status_code=404, detail="Association not found")
    
    # Get the item handler to check ownership
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check permissions: user must be owner of the item handler
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to modify this item handler")
    
    # Get the item to check ownership
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check permissions: user must be owner of the item (only owner can remove their item)
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Only item owner can remove their item")
    
    # Delete the association
    session.delete(association)
    session.commit()

    from app.services.agent.chat_runtime import invalidate_handler_cache

    invalidate_handler_cache(str(item_id))

    return {"message": "Item removed from item handler successfully"}


@router.get("/items/{item_handler_id}", response_model=List[Item])
def get_items_for_handler(
    *, session: SessionDep, current_user: CurrentUser, item_handler_id: uuid.UUID
) -> Any:
    """
    Get all items for an item handler.
    """
    # Check if item handler exists
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check permissions: user must be owner of the item handler or superuser
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to view this item handler")
    
    # Get the items
    statement = (
        select(Item)
        .join(ItemHandlerItem)
        .where(ItemHandlerItem.item_handler_id == item_handler_id)
    )
    items = session.exec(statement).all()
    
    return items


@router.get("/item-handlers/{item_id}", response_model=List[ItemHandler])
def get_handlers_for_item(
    *, session: SessionDep, current_user: CurrentUser, item_id: uuid.UUID
) -> Any:
    """
    Get all item handlers for an item.
    """
    # Check if item exists
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check permissions: user must be owner of the item or superuser
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to view this item")
    
    # Get the item handlers
    statement = (
        select(ItemHandler)
        .join(ItemHandlerItem)
        .where(ItemHandlerItem.item_id == item_id)
    )
    item_handlers = session.exec(statement).all()
    
    return item_handlers


# ItemHandlerUser routes
@router.post("/users", response_model=dict)
def add_user_to_handler(
    *, session: SessionDep, current_user: CurrentUser, request: AddUserToHandlerRequest
) -> Any:
    """
    Add a user to an item handler.
    """
    item_handler_id = request.item_handler_id
    user_id = request.user_id
    # Check if item handler exists
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check if user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check permissions: user must be owner of the item handler
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to modify this item handler")
    
    # Check if the association already exists
    existing_association = session.exec(
        select(ItemHandlerUser)
        .where(ItemHandlerUser.item_handler_id == item_handler_id)
        .where(ItemHandlerUser.user_id == user_id)
    ).first()
    
    if existing_association:
        raise HTTPException(status_code=400, detail="User is already associated with this item handler")
    
    # Create the association
    association = ItemHandlerUser(item_handler_id=item_handler_id, user_id=user_id)
    session.add(association)
    session.commit()
    
    return {"message": "User added to item handler successfully"}


@router.delete("/users/{item_handler_id}/{user_id}", response_model=dict)
def remove_user_from_handler(
    *, session: SessionDep, current_user: CurrentUser, item_handler_id: uuid.UUID, user_id: uuid.UUID
) -> Any:
    """
    Remove a user from an item handler.
    """
    # Check if the association exists
    association = session.exec(
        select(ItemHandlerUser)
        .where(ItemHandlerUser.item_handler_id == item_handler_id)
        .where(ItemHandlerUser.user_id == user_id)
    ).first()
    
    if not association:
        raise HTTPException(status_code=404, detail="Association not found")
    
    # Get the item handler to check ownership
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check permissions: user must be owner of the item handler
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to modify this item handler")
    
    # Delete the association
    session.delete(association)
    session.commit()
    
    return {"message": "User removed from item handler successfully"}


@router.get("/users/{item_handler_id}", response_model=List[User])
def get_users_for_handler(
    *, session: SessionDep, current_user: CurrentUser, item_handler_id: uuid.UUID
) -> Any:
    """
    Get all users for an item handler.
    """
    # Check if item handler exists
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    
    # Check permissions: user must be owner of the item handler or superuser
    if not current_user.is_superuser and (item_handler.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to view this item handler")
    
    # Get the users
    statement = (
        select(User)
        .join(ItemHandlerUser)
        .where(ItemHandlerUser.item_handler_id == item_handler_id)
    )
    users = session.exec(statement).all()
    
    return users


@router.get("/user-handlers/{user_id}", response_model=List[ItemHandler])
def get_handlers_for_user(
    *, session: SessionDep, current_user: CurrentUser, user_id: uuid.UUID
) -> Any:
    """
    Get all item handlers for a user.
    """
    # Check if user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check permissions: user must be viewing their own handlers or superuser
    if not current_user.is_superuser and (user_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions to view this user's handlers")
    
    # Get the item handlers
    statement = (
        select(ItemHandler)
        .join(ItemHandlerUser)
        .where(ItemHandlerUser.user_id == user_id)
    )
    item_handlers = session.exec(statement).all()
    
    return item_handlers
